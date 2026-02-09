-- Insert default settings if not exists
INSERT OR IGNORE INTO settings (id, provider, model, whisperModel, llamacppEndpoint)
VALUES ('1', 'llamacpp', 'LocalModel', 'large-v3', 'http://127.0.0.1:13141/v1');
