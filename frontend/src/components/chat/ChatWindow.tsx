import React, { useEffect, useState, useRef } from 'react';
import { useChatStore, ChatMessage } from '../../stores/chatStore';
import { Send, AlertTriangle, ShieldAlert, CheckCircle2, Sparkles, RefreshCw, Languages, Zap } from 'lucide-react';

interface ChatWindowProps {
  channelId: string;
  currentUserId: string;
  currentUserName: string;
}

export default function ChatWindow({ channelId, currentUserId, currentUserName }: ChatWindowProps) {
  const { messages, addMessage, updateMessageStatus, appendSuggestionChunk } = useChatStore();
  const [inputText, setInputText] = useState('');
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [isChecking, setIsChecking] = useState(false);
  const [interceptedModal, setInterceptedModal] = useState<{
    originalText: string;
    isToxic: boolean;
    severity: string;
    categories?: Record<string, number>;
    detectedLanguage?: string;
    suggestion: string;
    isStreamingSuggestion: boolean;
  } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pendingMsgIdRef = useRef<string | null>(null);
  const pendingTextRef = useRef<string>('');

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, interceptedModal]);

  // Connect to WebSocket
  useEffect(() => {
    const connectWs = () => {
      setWsStatus('connecting');
      const ws = new WebSocket('ws://localhost:8000/ws/chat');

      ws.onopen = () => {
        setWsStatus('connected');
        console.log('Connected to SafeChat ML WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);

          if (payload.type === 'classification') {
            setIsChecking(false);
            const data = payload.data;
            if (pendingTextRef.current) {
              // Pre-Send Gatekeeper mode
              if (data.is_toxic || data.severity === 'HIGH' || data.severity === 'MEDIUM') {
                setInterceptedModal({
                  originalText: pendingTextRef.current,
                  isToxic: true,
                  severity: data.severity,
                  categories: data.categories,
                  detectedLanguage: data.detected_language,
                  suggestion: data.suggestion || '',
                  isStreamingSuggestion: true,
                });
              } else {
                addMessage({
                  id: Date.now().toString(),
                  channelId,
                  senderId: currentUserId,
                  senderName: currentUserName,
                  content: pendingTextRef.current,
                  status: 'DELIVERED',
                  createdAt: new Date().toISOString(),
                  moderation: {
                    isToxic: false,
                    severity: 'SAFE',
                    detectedLanguage: data.detected_language,
                  }
                });
                pendingTextRef.current = '';
                setIsChecking(false);
              }
            } else if (pendingMsgIdRef.current) {
              let status: any = 'DELIVERED';
              if (data.severity === 'HIGH') status = 'BLOCKED';
              if (data.severity === 'MEDIUM') status = 'FLAGGED';
              updateMessageStatus(pendingMsgIdRef.current, status, {
                isToxic: data.is_toxic,
                severity: data.severity,
                detectedLanguage: data.detected_language,
                categories: data.categories,
                suggestion: '',
              });
            }
          } else if (payload.type === 'detox_start') {
            if (interceptedModal || pendingTextRef.current) {
              setInterceptedModal(prev => prev ? { ...prev, isStreamingSuggestion: true } : null);
            } else if (pendingMsgIdRef.current) {
              updateMessageStatus(pendingMsgIdRef.current, useChatStore.getState().messages.find(m => m.id === pendingMsgIdRef.current)?.status || 'FLAGGED', {
                isStreamingSuggestion: true,
              });
            }
          } else if (payload.type === 'detox_chunk') {
            if (interceptedModal || pendingTextRef.current) {
              setInterceptedModal(prev => prev ? { ...prev, suggestion: prev.suggestion + payload.data.chunk } : null);
            } else if (pendingMsgIdRef.current) {
              appendSuggestionChunk(pendingMsgIdRef.current, payload.data.chunk);
            }
          } else if (payload.type === 'detox_end') {
            setIsChecking(false);
            if (interceptedModal || pendingTextRef.current) {
              setInterceptedModal(prev => prev ? { ...prev, suggestion: payload.data.full_text, isStreamingSuggestion: false } : null);
            } else if (pendingMsgIdRef.current) {
              updateMessageStatus(pendingMsgIdRef.current, useChatStore.getState().messages.find(m => m.id === pendingMsgIdRef.current)?.status || 'FLAGGED', {
                suggestion: payload.data.full_text,
                isStreamingSuggestion: false,
              });
            }
          } else if (payload.type === 'new_message') {
            const msgData = payload.data;
            const isDup = useChatStore.getState().messages.some(m => m.content === msgData.content && m.senderId === msgData.sender_id && Math.abs(new Date(m.createdAt).getTime() - new Date(msgData.created_at || Date.now()).getTime()) < 5000);
            if (!isDup) {
              const mod = msgData.moderation || msgData.moderation_data;
              addMessage({
                id: msgData.id || msgData.message_id || Date.now().toString(),
                channelId: msgData.channel_id || channelId,
                senderId: msgData.sender_id,
                senderName: msgData.sender_name,
                content: msgData.content,
                status: msgData.status || 'DELIVERED',
                createdAt: msgData.created_at || new Date().toISOString(),
                moderation: mod ? {
                  isToxic: mod.is_toxic,
                  severity: mod.severity,
                  detectedLanguage: mod.detected_language,
                  categories: mod.categories,
                } : undefined
              });
            }
          }
        } catch (err) {
          console.error('WebSocket message error:', err);
        }
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        setIsChecking(false);
        // Try to reconnect after 3 seconds
        setTimeout(connectWs, 3000);
      };

      ws.onerror = () => {
        setWsStatus('disconnected');
        setIsChecking(false);
      };

      wsRef.current = ws;
    };

    connectWs();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  // Handle Send
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isChecking) return;

    const sentText = inputText.trim();
    pendingTextRef.current = sentText;
    if (interceptedModal) setInterceptedModal(null);
    setIsChecking(true);
    setTimeout(() => setIsChecking(false), 5000);
    setInputText('');

    // Try WebSocket send
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'message', text: sentText }));
    } else {
      // Fallback to HTTP POST if WebSocket is offline
      try {
        const res = await fetch('http://localhost:8000/api/v1/moderate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: sentText, channel_id: channelId, user_id: currentUserId })
        });
        const mlResult = await res.json();
        
        if (mlResult.is_toxic || mlResult.severity === 'HIGH' || mlResult.severity === 'MEDIUM') {
          setInterceptedModal({
            originalText: sentText,
            isToxic: true,
            severity: mlResult.severity,
            categories: mlResult.categories,
            detectedLanguage: mlResult.detected_language,
            suggestion: mlResult.suggestion || 'Please use polite language.',
            isStreamingSuggestion: false,
          });
          setIsChecking(false);
        } else {
          addMessage({
            id: Date.now().toString(),
            channelId,
            senderId: currentUserId,
            senderName: currentUserName,
            content: sentText,
            status: 'DELIVERED',
            createdAt: new Date().toISOString(),
            moderation: {
              isToxic: false,
              severity: 'SAFE',
              detectedLanguage: mlResult.detected_language,
            }
          });
          pendingTextRef.current = '';
          setIsChecking(false);
        }
      } catch (err) {
        console.error("Failed ML Call", err);
        addMessage({
          id: Date.now().toString(),
          channelId,
          senderId: currentUserId,
          senderName: currentUserName,
          content: sentText,
          status: 'DELIVERED',
          createdAt: new Date().toISOString(),
          moderation: { isToxic: false, severity: 'SAFE' }
        });
        pendingTextRef.current = '';
        setIsChecking(false);
      }
    }
  };

  const handleAcceptSuggestion = (msgId: string, suggestion: string) => {
    useChatStore.setState((state) => ({
      messages: state.messages.map((m) =>
        m.id === msgId
          ? {
              ...m,
              content: suggestion,
              status: 'DELIVERED',
              moderation: { ...m.moderation!, isToxic: false, severity: 'SAFE' }
            }
          : m
      )
    }));
  };

  const getLanguageBadge = (code?: string) => {
    if (!code) return null;
    const map: Record<string, string> = {
      'hi-en': '🇮🇳 Hinglish',
      'hi': '🇮🇳 Hindi',
      'en': '🇬🇧 English',
      'bn': '🇮🇳 Bengali',
      'ta': '🇮🇳 Tamil',
      'te': '🇮🇳 Telugu'
    };
    return (
      <span className="inline-flex items-center gap-1 text-[10px] bg-slate-200/80 text-slate-700 px-2 py-0.5 rounded-full font-medium">
        <Languages className="w-3 h-3" /> {map[code] || code.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      {/* WebSocket Status Bar */}
      <div className="bg-slate-900/90 backdrop-blur text-white text-xs py-1 px-4 flex items-center justify-between border-b border-slate-800 shadow-sm">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${
            wsStatus === 'connected' ? 'bg-emerald-400 animate-pulse' :
            wsStatus === 'connecting' ? 'bg-amber-400 animate-bounce' : 'bg-red-500'
          }`} />
          <span className="font-mono text-[11px]">
            {wsStatus === 'connected' ? '⚡ Real-time Hing-RoBERTa + HingGPT + Gemini WS Pipeline Active' :
             wsStatus === 'connecting' ? 'Connecting to ML Engine...' : 'Offline (HTTP Fallback mode)'}
          </span>
        </div>
      </div>

      {/* Messages List */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => {
          const isMe = msg.senderId === currentUserId;
          const isBlocked = msg.status === 'BLOCKED';
          const isFlagged = msg.status === 'FLAGGED';
          const mod = msg.moderation;

          return (
            <div key={msg.id} className={`flex flex-col max-w-[85%] ${isMe ? 'self-end items-end ml-auto' : 'self-start items-start'}`}>
              <div className="text-xs text-slate-400 mb-1 px-1 flex items-center gap-2">
                <span>{msg.senderName}</span>
                <span>•</span>
                <span>{new Date(msg.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                {getLanguageBadge(mod?.detectedLanguage)}
              </div>
              
              <div className={`relative p-4 rounded-2xl shadow-md border transition-all ${
                  isBlocked ? 'bg-gradient-to-r from-red-50/90 to-red-100/50 border-red-300 text-red-950 border-l-4 border-l-red-600' :
                  isFlagged ? 'bg-gradient-to-r from-amber-50/90 to-amber-100/50 border-amber-300 text-amber-950 border-l-4 border-l-amber-500' :
                  isMe ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white border-indigo-700 shadow-indigo-500/10' : 'bg-white border-slate-200/80 text-slate-800'
                }`}>
                
                {isBlocked ? (
                    <div className="flex flex-col gap-2.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 font-bold text-red-700 text-sm">
                            <ShieldAlert className="w-4 h-4 text-red-600" /> BLOCKED BY POLICY ({mod?.severity})
                          </div>
                          {mod?.categories && (
                            <div className="flex gap-1 flex-wrap">
                              {Object.entries(mod.categories)
                                .filter(([_, score]) => (score as number) > 0.4)
                                .map(([cat, score]) => (
                                  <span key={cat} className="text-[9px] bg-red-200/80 text-red-800 px-1.5 py-0.5 rounded font-mono uppercase font-semibold">
                                    {cat}: {((score as number)*100).toFixed(0)}%
                                  </span>
                                ))}
                            </div>
                          )}
                        </div>
                        <p className="italic text-red-800/70 text-sm line-through bg-red-100/50 p-2 rounded border border-red-200/60 font-mono">{msg.content}</p>
                    </div>
                ) : (
                    <div>
                        {isFlagged && (
                            <div className="flex items-center justify-between mb-2 pb-1 border-b border-amber-200/60">
                              <div className="flex items-center gap-1.5 font-bold text-amber-800 text-xs uppercase tracking-wide">
                                <AlertTriangle className="w-3.5 h-3.5 text-amber-600" /> FLAGGED BY MODEL ({mod?.severity})
                              </div>
                              {mod?.categories && (
                                <div className="flex gap-1 flex-wrap">
                                  {Object.entries(mod.categories)
                                    .filter(([_, score]) => (score as number) > 0.4)
                                    .map(([cat, score]) => (
                                      <span key={cat} className="text-[9px] bg-amber-200/80 text-amber-900 px-1.5 py-0.5 rounded font-mono uppercase font-semibold">
                                        {cat}: {((score as number)*100).toFixed(0)}%
                                      </span>
                                    ))}
                                </div>
                              )}
                            </div>
                        )}
                        <p className="leading-relaxed text-sm font-normal">{msg.content}</p>
                    </div>
                )}

                {/* AI Detox Suggestion Box */}
                {(isBlocked || isFlagged) && isMe && (
                  <div className="mt-3 bg-white/90 backdrop-blur p-3 rounded-xl border border-indigo-200/80 shadow-inner text-slate-800 text-sm">
                    <div className="flex items-center justify-between mb-1 text-xs font-semibold text-indigo-600">
                      <span className="flex items-center gap-1.5">
                        <Sparkles className="w-4 h-4 text-amber-500 animate-spin" style={{ animationDuration: '4s' }} /> 
                        Gemini AI Intent-Preserving Rewrite:
                      </span>
                      {mod?.isStreamingSuggestion && (
                        <span className="text-[10px] text-indigo-500 flex items-center gap-1 font-normal">
                          <RefreshCw className="w-3 h-3 animate-spin" /> Generating...
                        </span>
                      )}
                    </div>
                    <div className="font-medium text-slate-700 bg-indigo-50/50 p-2.5 rounded-lg border border-indigo-100/60 mt-1.5">
                      {mod?.suggestion ? (
                        <span>"{mod.suggestion}"</span>
                      ) : (
                        <span className="text-slate-400 italic">Thinking of a polite way to say this in {mod?.detectedLanguage === 'hi-en' ? 'Hinglish' : 'your language'}...</span>
                      )}
                    </div>
                    {mod?.suggestion && !mod.isStreamingSuggestion && (
                      <div className="mt-2.5 flex justify-end">
                        <button
                          onClick={() => handleAcceptSuggestion(msg.id, mod.suggestion!)}
                          className="text-xs bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white font-medium px-3 py-1.5 rounded-lg shadow-sm hover:shadow active:scale-95 transition-all flex items-center gap-1.5"
                        >
                          <Sparkles className="w-3.5 h-3.5 text-amber-300" /> Replace with this suggestion
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {msg.status === 'PENDING' && isMe && (
                  <div className="text-[10px] text-slate-400 mt-1 flex items-center gap-1 font-medium">
                      <div className="w-2 h-2 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div> Classifying with MuRIL...
                  </div>
              )}
              {msg.status === 'DELIVERED' && isMe && (
                  <div className="text-[10px] text-emerald-600 mt-1 flex items-center gap-1 font-medium">
                      <CheckCircle2 className="w-3 h-3" /> Verified Safe
                  </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* Sender-Side Pre-Send Interception Gatekeeper Modal */}
      {interceptedModal && (
        <div className="mx-6 mb-4 p-5 bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white rounded-2xl border border-indigo-500/40 shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-300 relative overflow-hidden z-20">
          <div className="absolute -top-24 -right-24 w-48 h-48 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none"></div>
          
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-indigo-800/60">
            <div className="flex items-center gap-2 font-bold text-amber-400 text-sm tracking-wide">
              <ShieldAlert className="w-5 h-5 text-red-400 animate-bounce" />
              <span>INTERCEPTED BEFORE SENDING (HING-ROBERTA GATEKEEPER)</span>
              <span className="text-[10px] bg-red-500/20 text-red-300 border border-red-500/40 px-2 py-0.5 rounded-full font-mono uppercase">
                {interceptedModal.severity} SEVERITY
              </span>
            </div>
            {getLanguageBadge(interceptedModal.detectedLanguage)}
          </div>

          <p className="text-xs text-slate-300 mb-2 font-medium">
            We stopped this message from being delivered because our Stage-1 classifier detected content violating community guidelines:
          </p>
          
          {interceptedModal.categories && (
            <div className="flex gap-1.5 flex-wrap mb-3">
              {Object.entries(interceptedModal.categories)
                .filter(([_, score]) => (score as number) > 0.35)
                .map(([cat, score]) => (
                  <span key={cat} className="text-[10px] bg-red-950/80 text-red-300 border border-red-800 px-2 py-0.5 rounded-md font-mono uppercase font-semibold flex items-center gap-1 shadow-sm">
                    ⚠️ {cat}: {((score as number) * 100).toFixed(0)}%
                  </span>
                ))}
            </div>
          )}

          <div className="bg-red-950/40 border border-red-900/60 p-3 rounded-xl mb-4 text-xs font-mono text-red-200/80 line-through">
            "{interceptedModal.originalText}"
          </div>

          <div className="bg-indigo-900/40 border border-indigo-500/50 p-4 rounded-xl shadow-inner relative">
            <div className="flex items-center justify-between mb-2 text-xs font-semibold text-indigo-300">
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-amber-300 animate-spin" style={{ animationDuration: '4s' }} />
                <span>✨ Stage-2 Hybrid Detox Rewrite ({interceptedModal.detectedLanguage === 'hi-en' || !interceptedModal.detectedLanguage?.includes('hi') ? 'Local HingGPT' : 'Gemini 2.0 Flash'}):</span>
              </span>
              {interceptedModal.isStreamingSuggestion && (
                <span className="text-[10px] text-amber-300 flex items-center gap-1 font-normal animate-pulse">
                  <RefreshCw className="w-3 h-3 animate-spin" /> Rewriting constructively...
                </span>
              )}
            </div>
            <div className="text-sm font-medium text-white bg-black/30 p-3 rounded-lg border border-indigo-500/30 min-h-[44px] flex items-center">
              {interceptedModal.suggestion ? (
                <span className="text-indigo-100">"{interceptedModal.suggestion}"</span>
              ) : (
                <span className="text-slate-400 italic text-xs">Crafting a polite alternative...</span>
              )}
            </div>
          </div>

          <div className="mt-4 flex items-center justify-end gap-2 pt-2 border-t border-indigo-800/40">
            <button
              onClick={() => {
                setInterceptedModal(null);
                pendingTextRef.current = '';
                setIsChecking(false);
              }}
              className="px-3 py-2 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl transition-all font-medium"
            >
              🗑️ Discard
            </button>
            <button
              onClick={() => {
                const orig = interceptedModal.originalText;
                setInterceptedModal(null);
                pendingTextRef.current = '';
                setIsChecking(false);
                setInputText(orig);
              }}
              className="px-3.5 py-2 text-xs bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all font-medium border border-slate-600 hover:border-slate-500 flex items-center gap-1.5 shadow-sm"
            >
              ✏️ Edit Manually
            </button>
            <button
              disabled={!interceptedModal.suggestion || interceptedModal.isStreamingSuggestion}
              onClick={() => {
                const safeRewrite = interceptedModal.suggestion;
                setInterceptedModal(null);
                pendingTextRef.current = '';
                setIsChecking(false);
                addMessage({
                  id: Date.now().toString(),
                  channelId,
                  senderId: currentUserId,
                  senderName: currentUserName,
                  content: safeRewrite,
                  status: 'DELIVERED',
                  createdAt: new Date().toISOString(),
                  moderation: { isToxic: false, severity: 'SAFE' }
                });
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'safe_message', text: safeRewrite, skip_moderation: true }));
                } else {
                  fetch('http://localhost:8000/api/v1/moderate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: safeRewrite, channel_id: channelId, user_id: currentUserId, skip_moderation: true })
                  }).catch(err => console.error(err));
                }
              }}
              className="px-4 py-2 text-xs bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-semibold shadow-lg hover:shadow-emerald-500/20 transition-all flex items-center gap-1.5 active:scale-95"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-300" /> ✨ Send Suggested Rewrite
            </button>
          </div>
        </div>
      )}

      {/* Input Bar */}
      <div className="p-4 bg-white border-t border-slate-200/80 shadow-lg">
        <form onSubmit={handleSend} className="flex items-center gap-3 relative max-w-4xl mx-auto">
          <input
            type="text"
            value={inputText}
            onChange={(e) => {
              setInputText(e.target.value);
              if (interceptedModal) setInterceptedModal(null);
            }}
            onFocus={() => {
              if (interceptedModal) setInterceptedModal(null);
            }}
            disabled={false}
            placeholder={isChecking ? "⚡ Checking content safety with Hing-RoBERTa..." : "Type a message... (Try Hinglish: 'tu bahut bada bewakoof hai')"}
            className="flex-1 py-3.5 px-5 bg-slate-100/80 border border-slate-200 focus:bg-white focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-200/50 disabled:cursor-not-allowed rounded-full text-sm outline-none transition-all placeholder:text-slate-400 font-normal shadow-inner"
          />
          <button
            type="submit"
            disabled={!inputText.trim() || isChecking}
            className="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed text-white p-3.5 rounded-full transition-all flex items-center justify-center shadow-md hover:shadow-indigo-500/20 active:scale-95 min-w-[48px]"
          >
            {isChecking ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Send className="w-5 h-5 -ml-0.5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
