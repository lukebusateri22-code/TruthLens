# 🧪 Test Images for AI Detection

## 📁 Files in this folder:

### 🤖 AI-Generated Images (Should detect as FAKE):
- `ai_generated_1.jpg` - AI-generated landscape
- `ai_generated_2.jpg` - Synthetic pattern

### 📸 Real Images (Should detect as AUTHENTIC):
- `real_photo_1.jpg` - Real mountain landscape
- `real_photo_2.jpg` - Real portrait photo
- `real_photo_3.jpg` - Real person outdoors
- `real_photo_4.jpg` - Real portrait

---

## 🎯 How to Test:

1. **Open the webapp**: http://localhost:8504
2. **Upload an image** from this folder
3. **Click "ANALYZE IMAGE"**
4. **Check the results**:
   - Look at the "🤖 AI-Generated" card
   - Should show "✅ REAL IMAGE" or "⚠️ AI-GENERATED"
   - Confidence percentage shows model certainty

---

## 📊 Expected Results:

| File | Expected Detection | Model |
|------|-------------------|-------|
| `ai_generated_1.jpg` | ⚠️ AI-GENERATED | 91.82% Acc |
| `ai_generated_2.jpg` | ⚠️ AI-GENERATED | 91.82% Acc |
| `real_photo_1.jpg` | ✅ REAL IMAGE | 91.82% Acc |
| `real_photo_2.jpg` | ✅ REAL IMAGE | 91.82% Acc |
| `real_photo_3.jpg` | ✅ REAL IMAGE | 91.82% Acc |
| `real_photo_4.jpg` | ✅ REAL IMAGE | 91.82% Acc |

---

## 🔍 What to Look For:

### In the Results:
1. **Final Verdict** - Top banner shows overall result
2. **AI-Generated Card** - Shows specific AI detection
3. **Confidence Score** - Higher = more certain
4. **Model Info** - Shows 91.82% accuracy

### Good Signs:
- ✅ Real photos detected as authentic
- ⚠️ AI images detected as AI-generated
- High confidence scores (>80%)

### If Results Are Wrong:
- Model may need more training data
- Some images are harder to classify
- Edge cases exist in any ML model

---

## 💡 Tips:

- Try uploading your own AI-generated images (Stable Diffusion, DALL-E, Midjourney)
- Compare confidence scores between real and AI images
- Test with different types of images (portraits, landscapes, abstract)
- The model was trained on 30K images with 91.82% validation accuracy

---

**Model Details:**
- Architecture: Custom CNN (TinyDetector)
- Training: 20K CIFAKE images
- Accuracy: 91.82%
- Training Time: 5.2 minutes
- Image Size: 96x96 pixels
