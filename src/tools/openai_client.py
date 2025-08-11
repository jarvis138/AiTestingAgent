import os
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

def generate_test_from_prompt(prompt: str) -> str:
    if not OPENAI_KEY:
        # fallback stub
        return """import { test, expect } from '@playwright/test';
test('generated from prompt', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example Domain/);
});"""
    try:
        import openai
        openai.api_key = OPENAI_KEY
    resp = openai.ChatCompletion.create(
      model=os.getenv('LLM_MODEL', 'gpt-4'),
      messages=[{'role':'user','content': prompt}],
      max_tokens=800,
      temperature=0.1
    )
    code = resp['choices'][0]['message']['content']
    return code
    except Exception as e:
        return f'/* OpenAI error: {e} */\n' + """import { test, expect } from '@playwright/test';
test('generated fallback', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example Domain/);
});"""
