use regex::Regex;

// Keep this first-layer deny-list short, readable, and easy to extend. These regexes
// are intentionally conservative; they are not a substitute for an OS sandbox.
pub const BLOCKED_COMMAND_PATTERNS: &[(&str, &str)] = &[
    (
        r"(?i)\brm\s+-(?:[^\s]*r[^\s]*f|[^\s]*f[^\s]*r)\s+(?:/|/\*|--no-preserve-root)",
        "recursive deletion of a filesystem root",
    ),
    (r"(?i)\bformat(?:\.com)?\s+[a-z]:", "formatting a drive"),
    (
        r"(?i)\b(?:diskpart|diskutil)\b.*\b(?:clean|eraseDisk|eraseVolume)\b",
        "erasing a disk or volume",
    ),
    (
        r"(?i)\bdd\s+.*\bof\s*=\s*/dev/(?:sd[a-z]|nvme|vd[a-z]|hd[a-z])",
        "writing directly to a disk device",
    ),
    (r"(?i)\bmkfs(?:\.\w+)?\b", "creating a filesystem"),
    (
        r"(?i)\b(?:shred|wipefs|sdelete(?:64)?)\b",
        "securely erasing files or disks",
    ),
    (
        r"(?i):\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        "a shell fork bomb",
    ),
    (
        r"(?i)\b(?:shutdown|reboot|poweroff)\b",
        "shutting down or restarting the system",
    ),
    (
        r"(?i)\bgit\s+clean\s+-[^\s]*f[^\s]*d",
        "forcibly deleting untracked repository files",
    ),
    (
        r"(?i)\bremove-item\b.*-(?:recurse|r)\b.*-(?:force|f)\b.*[a-z]:\\",
        "recursive forced deletion from a Windows drive",
    ),
    (
        r"(?i)\b(?:del|erase|rd|rmdir)\s+/[sq][^\r\n]*[a-z]:\\",
        "recursive deletion from a Windows drive",
    ),
];

pub fn blocked_command_reason(command: &str) -> Result<Option<&'static str>, String> {
    for (pattern, reason) in BLOCKED_COMMAND_PATTERNS {
        let matcher = Regex::new(pattern)
            .map_err(|error| format!("Invalid dangerous-command pattern '{pattern}': {error}"))?;
        if matcher.is_match(command) {
            return Ok(Some(reason));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{blocked_command_reason, BLOCKED_COMMAND_PATTERNS};

    #[test]
    fn rejects_dangerous_command_patterns() {
        for command in [
            "rm -rf /",
            "format C:",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sdb",
            ":(){ :|:& };:",
            "git clean -fdx",
        ] {
            assert!(
                blocked_command_reason(command)
                    .expect("patterns compile")
                    .is_some(),
                "expected command to be blocked: {command}"
            );
        }
    }

    #[test]
    fn permits_commands_outside_the_deny_list() {
        for command in ["cargo test", "git status", "rm -rf ./target"] {
            assert_eq!(
                blocked_command_reason(command).expect("patterns compile"),
                None,
                "expected command to remain approval-gated, not blocked: {command}"
            );
        }
        assert!(!BLOCKED_COMMAND_PATTERNS.is_empty());
    }
}
