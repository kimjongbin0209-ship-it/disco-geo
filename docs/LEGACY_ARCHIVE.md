# Legacy archive

**English** | [한국어](LEGACY_ARCHIVE_ko.md)

The previous object-recognition and TripoSplat study is preserved outside this active research checkout:

`/Users/jongbinkim/Documents/ChatGPT/disco-geo-legacy-20261028-7cb6a52`

Result commit: `78649360cb01b9d80056db4f0785c8e98b33bedd` (`result commit with readme`). It was `7cb6a52ba4517a9d549464d2be6ccc9e84cbbab6`, the hash in the archive path, until 2026-10-31: before the first push of this history, `runs/` and `datasets/` were removed from the unpushed commits because one data file exceeded GitHub's 100 MiB limit. The original commits remain in the local branch `server-backup-before-strip`, and the data itself is in the archive.

The archive contains the old source, documentation, datasets, GPU checkpoints, results, and the copied Linux `.venv.cuda`. `MANIFEST.json` records the exact moved paths and copied metadata checksums. Source paths were renamed on the same filesystem rather than discarded. The `.git` history and the usable macOS `.venv` were retained; the history was later rewritten only to drop those data folders, as described above.

The active tree has been rebuilt for limited-reward spatial ability evaluation with Memory Maze. Old result numbers do not describe this new study.
