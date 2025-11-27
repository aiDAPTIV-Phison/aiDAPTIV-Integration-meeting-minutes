-- Insert default settings if not exists
INSERT OR IGNORE INTO settings (id, provider, model, whisperModel, ollamaEndpoint, openaiCompatibleEndpoint, openaiCompatibleApiKey)
VALUES ('1', 'openai-compatible', 'LocalModel', 'large-v3', NULL, 'http://127.0.0.1:13141/v1', 'local');

