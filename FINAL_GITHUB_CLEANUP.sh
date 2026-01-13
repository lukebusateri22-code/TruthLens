#!/bin/bash

echo "🧹 Final GitHub Cleanup"
echo "======================"
echo ""

# Remove all redundant/temporary files
echo "📝 Removing redundant files..."
rm -f cleanup_for_github.sh
rm -f cleanup_redundant.sh
rm -f FINAL_CLEANUP.sh
rm -f CLEANED_PROJECT_SUMMARY.md
rm -f test_ensemble.py
rm -f test_samples.py
rm -f find_confident_samples.py
rm -f create_test_set.py
rm -f webapp/app_dark.py  # Keep only app.py

# Remove test folders (keep structure)
echo "📁 Cleaning test folders..."
rm -rf test_samples/
rm -rf best_examples/
# Keep demo_test_set for demos

# Remove cache and temp files
echo "🗑️  Removing cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

# Keep directory structure
echo "📂 Ensuring directory structure..."
mkdir -p data/train data/val data/test
touch data/train/.gitkeep data/val/.gitkeep data/test/.gitkeep 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Final file count:"
echo "   Python files: $(find . -name "*.py" -not -path "./venv/*" | wc -l | tr -d ' ')"
echo "   Documentation: $(find . -name "*.md" | wc -l | tr -d ' ')"
echo ""
echo "📁 Project structure:"
ls -1 | grep -v venv | head -20
echo ""
echo "🚀 Ready for GitHub!"
