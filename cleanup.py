"""
Clean up unnecessary documentation files
Keep: code, config, single README
Delete: redundant docs
"""

import os

# Files to delete (redundant documentation)
files_to_delete = [
    "QUICKSTART.md",
    "BUILD_SUMMARY.md",
    "COMPLETE_DELIVERY_REPORT.md",
    "DEPLOYMENT_GUIDE.md",
    "FINAL_VERIFICATION.md",
    "HOW_IT_WORKS.md",
    "INDEX.md",
    "RUN_AND_DEPLOY.md",
    "START_HERE.md",
    "VISUAL_GUIDE.md",
    "app_config.yaml",  # Only needed for HF Spaces, can recreate if needed
]

for file in files_to_delete:
    filepath = os.path.join(".", file)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted: {file}")
    else:
        print(f"Not found: {file}")

print("\n✓ Cleanup complete!")
print("\nRemaining files:")
for file in sorted(os.listdir(".")):
    if not file.startswith("."):
        print(f"  {file}")
