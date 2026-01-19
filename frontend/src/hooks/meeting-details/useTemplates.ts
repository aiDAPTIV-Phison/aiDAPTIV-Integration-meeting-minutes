import { useState, useEffect, useCallback } from 'react';
import { invoke as invokeTauri } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import Analytics from '@/lib/analytics';

interface TemplateLanguagePreference {
  selectedTemplate: string;
  selectedLanguage: string;
}

const DEFAULT_PREFERENCE: TemplateLanguagePreference = {
  selectedTemplate: 'brief_summary',
  selectedLanguage: 'en',
};

// Load template and language preferences from localStorage
const loadPreferenceFromStorage = (): TemplateLanguagePreference => {
  if (typeof window === 'undefined') {
    return DEFAULT_PREFERENCE;
  }

  try {
    const stored = localStorage.getItem('templateLanguagePreference');
    if (stored) {
      const parsed = JSON.parse(stored);
      // Validate the structure
      if (parsed && typeof parsed === 'object') {
        return {
          selectedTemplate: parsed.selectedTemplate ?? DEFAULT_PREFERENCE.selectedTemplate,
          selectedLanguage: parsed.selectedLanguage ?? DEFAULT_PREFERENCE.selectedLanguage,
        };
      }
    }
  } catch (error) {
    console.error('Failed to load template/language preference from localStorage:', error);
  }

  return DEFAULT_PREFERENCE;
};

export function useTemplates() {
  const [availableTemplates, setAvailableTemplates] = useState<Array<{
    id: string;
    name: string;
    description: string;
  }>>([]);

  // Initialize from localStorage
  const initialPreference = loadPreferenceFromStorage();
  const [selectedTemplate, setSelectedTemplate] = useState<string>(initialPreference.selectedTemplate);
  const [selectedLanguage, setSelectedLanguage] = useState<string>(initialPreference.selectedLanguage);

  // Save preferences to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const preference: TemplateLanguagePreference = {
          selectedTemplate,
          selectedLanguage,
        };
        localStorage.setItem('templateLanguagePreference', JSON.stringify(preference));
        console.log('💾 Saved template/language preference to localStorage:', preference);
      } catch (error) {
        console.error('Failed to save template/language preference to localStorage:', error);
      }
    }
  }, [selectedTemplate, selectedLanguage]);

  // Fetch available templates on mount
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const templates = await invokeTauri('api_list_templates') as Array<{
          id: string;
          name: string;
          description: string;
        }>;
        console.log('Available templates:', templates);
        setAvailableTemplates(templates);
      } catch (error) {
        console.error('Failed to fetch templates:', error);
      }
    };
    fetchTemplates();
  }, []);

  // Handle template selection
  const handleTemplateSelection = useCallback((templateId: string, templateName: string) => {
    setSelectedTemplate(templateId);
    toast.success('Template selected', {
      description: `Using "${templateName}" template for summary generation`,
    });
    Analytics.trackFeatureUsed('template_selected');
  }, []);

  // Handle language selection
  const handleLanguageSelection = useCallback((languageCode: string) => {
    setSelectedLanguage(languageCode);
    toast.success('Language selected', {
      description: `Output language set to ${languageCode}`,
    });
  }, []);

  return {
    availableTemplates,
    selectedTemplate,
    selectedLanguage,
    handleTemplateSelection,
    handleLanguageSelection,
  };
}
