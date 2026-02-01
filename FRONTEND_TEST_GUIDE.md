# TruthLens Frontend Testing Guide

## 🎯 Test Images Location

**Path:** `test_images/`

### Authentic Images (10):
- `test_images/authentic/authentic_1.jpg` through `authentic_10.jpg`
- Expected Result: Should detect as **AUTHENTIC**

### Manipulated Images (10):
- `test_images/manipulated/manipulated_1.jpg` through `manipulated_10.jpg` (or .tif)
- Expected Result: Should detect as **MANIPULATED**

---

## 🧪 Testing Checklist

### 1. Manipulation Detection Test
- [ ] Upload `authentic_1.jpg` → Should show "AUTHENTIC" with confidence
- [ ] Upload `manipulated_1.jpg` → Should show "MANIPULATED" with confidence
- [ ] Test all 20 images
- [ ] Verify confidence scores are reasonable (>70%)
- [ ] Check processing time (<5 seconds per image)

### 2. AI Detection Test (if available)
- [ ] Upload AI-generated image → Should detect as "AI-GENERATED"
- [ ] Upload real photo → Should detect as "REAL"

### 3. Deepfake Detection Test (if available)
- [ ] Upload deepfake image → Should detect as "DEEPFAKE"
- [ ] Upload real face → Should detect as "REAL"

### 4. UI/UX Test
- [ ] All three detectors load without errors
- [ ] Model info displays correctly:
  - Manipulation: 79.87% validation, 91% real-world
  - AI Detection: 87% real-world
  - Deepfake: 88.5% validation
- [ ] Confidence bars display properly
- [ ] Results are clearly visible
- [ ] No console errors

### 5. Performance Test
- [ ] Single image upload works
- [ ] Multiple images can be tested in sequence
- [ ] No memory leaks (test 10+ images)
- [ ] Page doesn't crash or freeze

---

## 📊 Expected Results

### Model Performance:
- **Manipulation Detection:** 91% overall (82% authentic, 100% manipulated)
- **AI Detection:** 87% real-world accuracy
- **Deepfake Detection:** 88.5% validation accuracy

### Test Set Performance (Expected):
- Authentic: ~8-9 out of 10 correct
- Manipulated: ~9-10 out of 10 correct

---

## 🐛 Known Issues (if any)

None currently - all models loading successfully!

---

## ✅ Success Criteria

- ✅ All models load without errors
- ✅ Predictions are consistent with expected performance
- ✅ UI is responsive and professional
- ✅ No crashes or freezes
- ✅ Processing time <5 seconds per image

---

## 🚀 How to Test

1. **Start the webapp:**
   ```bash
   streamlit run webapp/app.py --server.port 8501
   ```

2. **Open browser:**
   - Navigate to `http://localhost:8501`

3. **Test each image:**
   - Click "Browse files" or drag & drop
   - Select image from `test_images/` folder
   - Wait for prediction
   - Verify result matches expected outcome

4. **Record results:**
   - Note any incorrect predictions
   - Check confidence scores
   - Monitor processing time

---

## 📝 Test Results Template

```
Date: ___________
Tester: ___________

Authentic Images (10):
- authentic_1.jpg: ✅/❌ (Confidence: __%)
- authentic_2.jpg: ✅/❌ (Confidence: __%)
- authentic_3.jpg: ✅/❌ (Confidence: __%)
- authentic_4.jpg: ✅/❌ (Confidence: __%)
- authentic_5.jpg: ✅/❌ (Confidence: __%)
- authentic_6.jpg: ✅/❌ (Confidence: __%)
- authentic_7.jpg: ✅/❌ (Confidence: __%)
- authentic_8.jpg: ✅/❌ (Confidence: __%)
- authentic_9.jpg: ✅/❌ (Confidence: __%)
- authentic_10.jpg: ✅/❌ (Confidence: __%)

Manipulated Images (10):
- manipulated_1: ✅/❌ (Confidence: __%)
- manipulated_2: ✅/❌ (Confidence: __%)
- manipulated_3: ✅/❌ (Confidence: __%)
- manipulated_4: ✅/❌ (Confidence: __%)
- manipulated_5: ✅/❌ (Confidence: __%)
- manipulated_6: ✅/❌ (Confidence: __%)
- manipulated_7: ✅/❌ (Confidence: __%)
- manipulated_8: ✅/❌ (Confidence: __%)
- manipulated_9: ✅/❌ (Confidence: __%)
- manipulated_10: ✅/❌ (Confidence: __%)

Overall Accuracy: __/20 (__%)

Notes:
_______________________________________________
_______________________________________________
```
