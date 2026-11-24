//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of Grand Theft Auto VI.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = explicit_config_path(|k| std::env::var_os(k));
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed.paths, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FilePaths,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env_dir(&env, k);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => file.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.cache_dir)
            .map_or_else(|| default_cache_dir(&env), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_data_dir(&env), Ok)?;
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            source,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

/// The config file explicitly named by the environment, if any. An empty
/// `VICTORY_CONFIG` means unset, like the other path variables.
fn explicit_config_path(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    env("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// A non-empty environment value, as a path. An empty variable counts as unset.
fn env_dir(env: &dyn Fn(&str) -> Option<OsString>, key: &str) -> Option<PathBuf> {
    env(key).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `%LOCALAPPDATA%/victory/cache` on Windows; else XDG cache or home cache.
fn default_cache_dir(env: &dyn Fn(&str) -> Option<OsString>) -> Result<PathBuf> {
    let base = env_dir(env, "LOCALAPPDATA")
        .or_else(|| env_dir(env, "XDG_CACHE_HOME"))
        .or_else(|| env_dir(env, "HOME").map(|h| h.join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join("cache"))
}

/// `%LOCALAPPDATA%/victory/data` on Windows; `$XDG_DATA_HOME` else
/// `$HOME/.local/share` elsewhere. Deliberately never under the cache root.
fn default_data_dir(env: &dyn Fn(&str) -> Option<OsString>) -> Result<PathBuf> {
    let base = env_dir(env, "LOCALAPPDATA")
        .or_else(|| env_dir(env, "XDG_DATA_HOME"))
        .or_else(|| env_dir(env, "HOME").map(|h| h.join(".local").join("share")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join("data"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn data_dir_not_under_cache_without_localappdata() {
        let env = |k: &str| match k {
            "LOCALAPPDATA" => None,
            "XDG_CACHE_HOME" => Some(OsString::from("/tmp/probe-home/.cache")),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        let cache = default_cache_dir(&env).unwrap();
        let data = default_data_dir(&env).unwrap();
        assert_eq!(cache, PathBuf::from("/tmp/probe-home/.cache/victory/cache"));
        assert_eq!(
            data,
            PathBuf::from("/tmp/probe-home/.local/share/victory/data")
        );
        assert!(!data.starts_with("/tmp/probe-home/.cache"));
    }

    #[test]
    fn empty_localappdata_falls_back_to_xdg() {
        let env = |k: &str| match k {
            "LOCALAPPDATA" => Some(OsString::new()),
            "XDG_CACHE_HOME" => Some(OsString::from("/tmp/probe-xdg/cache")),
            "XDG_DATA_HOME" => Some(OsString::from("/tmp/probe-xdg/data")),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        assert_eq!(
            default_cache_dir(&env).unwrap(),
            PathBuf::from("/tmp/probe-xdg/cache/victory/cache")
        );
        assert_eq!(
            default_data_dir(&env).unwrap(),
            PathBuf::from("/tmp/probe-xdg/data/victory/data")
        );
    }

    #[test]
    fn xdg_home_without_home() {
        let env = |k: &str| match k {
            "XDG_CACHE_HOME" => Some(OsString::from("/tmp/probe-xdg/cache")),
            "XDG_DATA_HOME" => Some(OsString::from("/tmp/probe-xdg/data")),
            _ => None,
        };
        assert_eq!(
            default_cache_dir(&env).unwrap(),
            PathBuf::from("/tmp/probe-xdg/cache/victory/cache")
        );
        assert_eq!(
            default_data_dir(&env).unwrap(),
            PathBuf::from("/tmp/probe-xdg/data/victory/data")
        );
    }

    #[test]
    fn empty_home_is_not_a_relative_path() {
        let env = |k: &str| match k {
            "HOME" => Some(OsString::new()),
            _ => None,
        };
        assert!(default_cache_dir(&env).is_err());
        assert!(default_data_dir(&env).is_err());
    }

    #[test]
    fn environment_overrides_file() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            cache_dir: Some("cache-from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(OsString::from("from-env")),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_GAME_DIR" => Some(OsString::new()),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FilePaths {
            test_titles: vec!["from-file".into()],
            ..Default::default()
        };
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(OsString::new()),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FilePaths::default();
        let env = |k: &str| match k {
            "VICTORY_TEST_TITLES" => Some(value.clone()),
            "HOME" => Some(OsString::from("/tmp/probe-home")),
            _ => None,
        };
        let cfg = Config::resolve(file, None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn empty_config_env_is_ignored() {
        let empty = |k: &str| (k == "VICTORY_CONFIG").then(OsString::new);
        assert_eq!(explicit_config_path(empty), None);
        let set = |k: &str| (k == "VICTORY_CONFIG").then(|| OsString::from("custom.toml"));
        assert_eq!(
            explicit_config_path(set),
            Some(PathBuf::from("custom.toml"))
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }
}
