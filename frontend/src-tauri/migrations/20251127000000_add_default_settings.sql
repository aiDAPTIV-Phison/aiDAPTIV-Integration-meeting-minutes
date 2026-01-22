-- Insert default settings if not exists
INSERT OR IGNORE INTO settings (id, provider, model, whisperModel)
VALUES ('1', 'ollama', 'llama3.2:latest', 'large-v3');

