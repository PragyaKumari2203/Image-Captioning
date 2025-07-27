import React from 'react';

export default function Footer() {
  return (
    <footer className="relative z-10 bg-gradient-to-r from-gray-900 via-indigo-950 to-gray-900 text-gray-300 border-t border-gray-800 px-6 py-6 text-center text-sm sm:text-base overflow-hidden">
      {/* Ambient Glow Animation */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <div className="absolute -bottom-10 left-1/4 w-72 h-72 bg-purple-800 opacity-20 blur-3xl animate-pulse" />
        <div className="absolute -top-10 right-1/4 w-64 h-64 bg-indigo-700 opacity-10 blur-2xl animate-ping" />
      </div>

      {/* Footer Content */}
      <div className="relative z-10">
        <p className="mb-2">
          © {new Date().getFullYear()} <span className="font-semibold text-white">Visual Description System</span>
        </p>
        <p className="text-gray-400">
          Built with <span className="animate-pulse text-red-400">❤️</span> 
        </p>
      </div>
    </footer>
  );
}
