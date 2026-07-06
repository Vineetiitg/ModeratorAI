import React from 'react';
import ChatWindow from './components/chat/ChatWindow';

function App() {
  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-xl shadow-xl overflow-hidden flex h-[80vh]">
        
        {/* Sidebar */}
        <div className="w-64 bg-slate-900 text-white flex flex-col">
          <div className="p-4 border-b border-slate-700">
            <h1 className="text-xl font-bold flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
              SafeChat
            </h1>
            <p className="text-xs text-slate-400 mt-1">Real-time ML Moderation</p>
          </div>
          <div className="p-4 flex-1">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Channels</h2>
            <ul className="space-y-1">
              <li className="bg-slate-800 text-white px-3 py-2 rounded-lg cursor-pointer flex items-center gap-2 transition-colors">
                <span className="text-slate-400">#</span> general
              </li>
            </ul>
          </div>
          <div className="p-4 border-t border-slate-700 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center font-bold">
                V
              </div>
              <span>Vineet</span>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col bg-white relative">
          <div className="p-4 border-b border-slate-200 shadow-sm z-10 bg-white">
            <h2 className="font-semibold text-slate-800 text-lg"># general</h2>
            <p className="text-xs text-slate-500">Welcome to general discussion.</p>
          </div>
          
          <div className="flex-1 overflow-hidden relative">
            <ChatWindow channelId="general" currentUserId="user-123" currentUserName="Vineet" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
