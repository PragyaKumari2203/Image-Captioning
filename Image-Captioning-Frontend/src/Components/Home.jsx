import { motion } from "framer-motion";
import React from "react";

export default function Home() {
  return (
    <div className="relative min-h-screen bg-gradient-to-tr from-black via-gray-900 to-indigo-950 flex items-center justify-center text-center px-4 py-16"> 
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute top-0 left-1/3 w-96 h-96 bg-purple-700 opacity-20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-indigo-500 opacity-10 rounded-full blur-2xl animate-pulse delay-500" />
      </div>

      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 bg-gradient-to-br from-white/5 via-white/10 to-white/5 backdrop-blur-2xl border border-white/10 shadow-2xl rounded-3xl px-10 py-16 max-w-2xl w-full" 
      >
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-500 via-purple-400 to-indigo-400 mb-8 leading-tight"> 
          Visual Description System
        </h1>
        <p className="text-gray-300 text-lg mb-8">
          Just upload an image — our model will do the talking.
        </p>
        <a
          href="/caption"
          className="inline-block bg-indigo-500 hover:bg-indigo-600 text-white px-8 py-3 rounded-full font-medium shadow-lg hover:shadow-xl transition-all duration-300"
        >
          Try It Now
        </a>
      </motion.div>
    </div>
  );
}