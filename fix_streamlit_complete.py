"""
Complete Streamlit Compatibility Fix
Removes ALL problematic width parameters from app_enhanced.py
"""

import re

print("="*70)
print("🔧 COMPLETE STREAMLIT COMPATIBILITY FIX")
print("="*70)

# Read the file
try:
    with open('src/app_enhanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    print("✅ File loaded successfully")
except FileNotFoundError:
    print("❌ Error: src/app_enhanced.py not found!")
    print("   Make sure you're in the project root directory")
    exit(1)

original_content = content
fixes_applied = []

# Fix 1: Remove use_container_width from st.image
pattern1 = r',\s*use_container_width\s*=\s*True'
if re.search(pattern1, content):
    content = re.sub(pattern1, '', content)
    fixes_applied.append("Removed use_container_width from st.image")

# Fix 2: Remove use_column_width from st.image
pattern2 = r',\s*use_column_width\s*=\s*True'
if re.search(pattern2, content):
    content = re.sub(pattern2, '', content)
    fixes_applied.append("Removed use_column_width from st.image")

# Fix 3: Remove width parameters that come before other params
pattern3 = r'use_container_width\s*=\s*True\s*,'
if re.search(pattern3, content):
    content = re.sub(pattern3, '', content)
    fixes_applied.append("Removed use_container_width before commas")

pattern4 = r'use_column_width\s*=\s*True\s*,'
if re.search(pattern4, content):
    content = re.sub(pattern4, '', content)
    fixes_applied.append("Removed use_column_width before commas")

# Fix 4: Clean up any standalone occurrences
content = content.replace('use_container_width=True', '')
content = content.replace('use_column_width=True', '')

# Fix 5: Clean up double commas that might result
content = content.replace(',,', ',')
content = content.replace(', )', ')')
content = content.replace(',  )', ')')

# Fix 6: Clean up trailing commas before closing parenthesis
content = re.sub(r',(\s*)\)', r'\1)', content)

# Write the fixed content
if content != original_content:
    with open('src/app_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*70)
    print("✅ FIXES APPLIED:")
    print("="*70)
    for i, fix in enumerate(fixes_applied, 1):
        print(f"  {i}. {fix}")
    
    print("\n" + "="*70)
    print("🎉 SUCCESS! File fixed and saved.")
    print("="*70)
    print("\nNext steps:")
    print("  1. Close any running Streamlit windows")
    print("  2. Run: streamlit run src/app_enhanced.py")
    print("  3. Test all tabs to verify everything works")
    print("="*70)
else:
    print("\n✅ No fixes needed - file is already compatible!")

print("\nDone! ✨")