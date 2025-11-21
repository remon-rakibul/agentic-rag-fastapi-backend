# Prompts Quick Reference

## 📝 How to Modify Prompts

### Step 1: Edit `prompts.json`
```bash
nano refactored/app/workflows/prompts.json
# or
vim refactored/app/workflows/prompts.json
```

### Step 2: Modify Any Prompt
```json
{
  "prompts": {
    "generate_answer": {
      "template": "YOUR CUSTOM PROMPT HERE\nQuestion: {question}\nContext: {context}"
    }
  }
}
```

### Step 3: Restart Server
```bash
# Stop server (Ctrl+C)
python run.py
```

**Done!** ✅

---

## 🎯 Common Modifications

### Make Answers More Detailed
```json
"generate_answer": {
  "template": "Provide a comprehensive, detailed answer.\nQuestion: {question}\nContext: {context}\n\nAnswer:"
}
```

### Change Model
```json
"settings": {
  "default_model": "gpt-4o"
}
```

### Adjust Retrieval Count
```json
"settings": {
  "default_k": 10
}
```

### Customize System Message
```json
"system_messages": {
  "generate_query_or_respond": {
    "template": "Your custom system message here. Current question: '{current_question}'"
  }
}
```

---

## 📋 Available Prompts

| Prompt Name | Variables | Purpose |
|------------|-----------|---------|
| `grade_documents` | `{question}`, `{context}` | Check if docs are relevant |
| `rewrite_question` | `{question}` | Improve question for search |
| `generate_answer` | `{question}`, `{context}` | Generate final answer |
| `generate_query_or_respond` | `{current_question}` | System message for LLM |

---

## ⚠️ Important Notes

- **Always include variables**: `{question}`, `{context}`, `{current_question}`
- **Restart required**: Changes take effect after server restart
- **JSON syntax**: Must be valid JSON (use a validator if unsure)
- **Backup first**: Copy `prompts.json` before major changes

---

## 🐛 Troubleshooting

**Error: "Prompts file not found"**
→ Check file exists at `app/workflows/prompts.json`

**Error: "Invalid JSON"**
→ Validate JSON syntax (missing comma, quote, etc.)

**Changes not applying**
→ Restart the server

---

**File Location:** `refactored/app/workflows/prompts.json`  
**Documentation:** `refactored/app/workflows/PROMPTS_GUIDE.md`

