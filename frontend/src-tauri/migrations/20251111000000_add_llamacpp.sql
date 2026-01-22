-- Add Llama.cpp endpoint and API key columns to settings table
ALTER TABLE settings ADD COLUMN llamacppEndpoint TEXT;
ALTER TABLE settings ADD COLUMN llamacppApiKey TEXT;
