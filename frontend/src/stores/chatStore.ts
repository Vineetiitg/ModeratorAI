import { create } from 'zustand';

export interface ChatMessage {
  id: string;
  channelId: string;
  senderId: string;
  senderName: string;
  content: string;
  status: 'PENDING' | 'DELIVERED' | 'BLOCKED' | 'FLAGGED';
  moderation?: {
    isToxic: boolean;
    severity: string;
    suggestion?: string;
    isStreamingSuggestion?: boolean;
    detectedLanguage?: string;
    categories?: Record<string, float | number>;
  };
  createdAt: string;
}

interface ChatState {
  messages: ChatMessage[];
  addMessage: (msg: ChatMessage) => void;
  updateMessageStatus: (id: string, status: any, moderation: any) => void;
  appendSuggestionChunk: (id: string, chunk: string) => void;
  setMessages: (msgs: ChatMessage[]) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [
    {
      id: 'demo-1',
      channelId: 'general',
      senderId: 'user-hr',
      senderName: 'Aditi (HR Lead)',
      content: 'Welcome team! Please remember our community guidelines: keep discussions constructive and respectful across all languages.',
      status: 'DELIVERED',
      createdAt: new Date(Date.now() - 300000).toISOString(),
      moderation: { isToxic: false, severity: 'SAFE', detectedLanguage: 'en' }
    },
    {
      id: 'demo-2',
      channelId: 'general',
      senderId: 'user-dev',
      senderName: 'Rahul (Tech Tech)',
      content: 'Got it Aditi! We just deployed the new Hybrid Moderation Engine (Hing-RoBERTa + HingGPT + Gemini 2.0). Try testing it out!',
      status: 'DELIVERED',
      createdAt: new Date(Date.now() - 120000).toISOString(),
      moderation: { isToxic: false, severity: 'SAFE', detectedLanguage: 'en' }
    }
  ],
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  setMessages: (msgs) => set({ messages: msgs }),
  updateMessageStatus: (id, status, moderation) => set((state) => ({
    messages: state.messages.map((m) => 
      m.id === id ? { ...m, status, moderation: { ...m.moderation, ...moderation } } : m
    )
  })),
  appendSuggestionChunk: (id, chunk) => set((state) => ({
    messages: state.messages.map((m) => {
      if (m.id !== id) return m;
      const currentSuggestion = m.moderation?.suggestion || '';
      return {
        ...m,
        moderation: {
          ...m.moderation!,
          suggestion: currentSuggestion + chunk,
          isStreamingSuggestion: true,
        }
      };
    })
  })),
}));

