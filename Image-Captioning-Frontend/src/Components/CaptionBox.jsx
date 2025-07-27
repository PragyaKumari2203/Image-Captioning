import { motion } from "framer-motion";
import { TypeAnimation } from "react-type-animation";
import { useState } from "react";
import { FaUpload, FaCheck, FaMagic } from "react-icons/fa";

export default function CaptionBox() {
  const [caption, setCaption] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState("");

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setImage(URL.createObjectURL(file));
  };

  const generateCaption = async () => {
    if (!image) return;
  
    setIsLoading(true);
    setCaption(null);
  
    try {
      const file = await fetch(image)
        .then((r) => r.blob())
        .then(
          (blobFile) => new File([blobFile], fileName, { type: blobFile.type })
        );
  
      const formData = new FormData();
      formData.append("image", file);
      const response = await fetch("http://127.0.0.1:8000/caption", {
        method: "POST",
        mode: "cors",
        credentials: "include",
        headers: {
          Accept: "application/json",
        },
        body: formData,
      });
  
      console.log("Raw response:", response); // Debug log
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const data = await response.json(); 
      console.log("Response data:", data); // Debug log
  
      if (data.caption) {
        setCaption(data.caption);
      } else {
        throw new Error("No caption in response");
      }
  
    } catch (error) {
      console.error("Error generating caption:", error);
      setCaption("Failed to generate caption. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const dotVariants = {
    blink: {
      opacity: [0.3, 1, 0.3],
      scale: [0.8, 1.2, 0.8],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut",
      },
    },
  };

  const LoadingDots = () => (
    <div className="flex items-center">
      <motion.div
        variants={dotVariants}
        animate="blink"
        className="w-3 h-3 rounded-full bg-cyan-400 mr-2"
      />
      <motion.div
        variants={dotVariants}
        animate="blink"
        style={{ animationDelay: "0.2s" }}
        className="w-3 h-3 rounded-full bg-purple-400 mr-2"
      />
      <motion.div
        variants={dotVariants}
        animate="blink"
        style={{ animationDelay: "0.4s" }}
        className="w-3 h-3 rounded-full bg-pink-400 mr-4"
      />
    </div>
  );

  const StaticDots = () => (
    <div className="flex items-center">
      <div className="w-3 h-3 rounded-full bg-cyan-400 opacity-80 mr-2" />
      <div className="w-3 h-3 rounded-full bg-purple-400 opacity-80 mr-2" />
      <div className="w-3 h-3 rounded-full bg-pink-400 opacity-80 mr-4" />
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="relative min-h-screen p-16 bg-[#0a0a0a] overflow-hidden"
    >
      {/* Background & Borders */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-indigo-900/20 rounded-full blur-[100px] animate-float-slow"></div>
        <div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-purple-900/15 rounded-full blur-[80px] animate-float-medium"></div>
        <div className="absolute top-1/2 right-1/4 w-64 h-64 bg-cyan-900/10 rounded-full blur-[60px] animate-float-fast"></div>
      </div>

      <div className="absolute inset-0 rounded-xl pointer-events-none">
        <div className="absolute inset-0 border border-white/5 rounded-xl"></div>
        <div className="absolute inset-0 border border-indigo-500/10 rounded-xl shadow-lg shadow-indigo-500/5"></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-4xl mx-auto flex flex-col items-center justify-center min-h-[90vh] py-8">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-center mb-8 pb-4"
        >
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 mb-4 leading-tight">
            Visual Description System
          </h2>
          <p className="text-lg text-gray-400 max-w-2xl">
            Transform your images into narratives with AI-powered captioning
          </p>
        </motion.div>

        {/* Upload Card */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="w-full max-w-2xl bg-gray-900/50 backdrop-blur-lg border border-gray-800 rounded-2xl p-8 shadow-xl mb-12"
        >
          <div className="text-center mb-4">
            <h2 className="text-xl font-bold text-white">Image Uploader</h2>
            <p className="text-gray-300 text-sm">
              Upload an image to generate caption
            </p>
          </div>

          {/* File Upload */}
          <label className="flex items-center justify-between p-3 border border-gray-600 rounded-lg bg-gray-900/40 hover:bg-gray-900/60 transition-colors cursor-pointer mb-4">
            <div className="flex items-center space-x-3">
              {image ? (
                <FaCheck className="text-green-400" />
              ) : (
                <FaUpload className="text-purple-400" />
              )}
              <span className="text-gray-200 text-sm truncate max-w-[180px]">
                {fileName || "Choose an image..."}
              </span>
            </div>
            <div className="px-3 py-1 bg-gray-700 rounded-md text-xs text-gray-300">
              Browse
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleUpload}
              className="hidden"
            />
          </label>

          {/* Image Preview */}
          {image && (
            <div className="flex flex-col items-center gap-4">
              <div className="relative max-w-xs">
                <img
                  src={image}
                  alt="Uploaded preview"
                  className="max-h-60 rounded-lg object-contain"
                />
                {fileName && (
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                    <span className="text-white text-xs truncate block">
                      {fileName}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </motion.div>

        {/* Generate Button Section */}
        {image && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mb-16"
          >
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={generateCaption}
              disabled={isLoading}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                isLoading
                  ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-purple-800 to-gray-900 text-white hover:shadow-xl hover:shadow-gray-500/10"
              }`}
            >
              <FaMagic />
              {isLoading ? "Generating..." : "Generate Caption"}
            </motion.button>
          </motion.div>
        )}

        {/* Caption Display Box */}
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="w-full max-w-3xl bg-gradient-to-br from-gray-900 to-gray-950 border border-gray-800 rounded-xl p-8 shadow-2xl overflow-hidden"
        >
          <div className="flex items-center mb-6">
            {isLoading ? <LoadingDots /> : <StaticDots />}
            <h3 className="text-xl font-medium text-gray-300">
              AI-Generated Caption
            </h3>
          </div>

          <div className="min-h-40 p-6 bg-gray-950/70 border border-gray-800 rounded-lg">
            {isLoading ? (
              <div className="flex items-center space-x-3 text-gray-500">
                <div className="flex space-x-1">
                  <motion.div
                    variants={dotVariants}
                    animate="blink"
                    className="w-2 h-2 rounded-full bg-cyan-400"
                  />
                  <motion.div
                    variants={dotVariants}
                    animate="blink"
                    style={{ animationDelay: "0.2s" }}
                    className="w-2 h-2 rounded-full bg-purple-400"
                  />
                  <motion.div
                    variants={dotVariants}
                    animate="blink"
                    style={{ animationDelay: "0.4s" }}
                    className="w-2 h-2 rounded-full bg-pink-400"
                  />
                </div>
                <span>Generating your magical caption...</span>
              </div>
            ) : caption ? (
              <TypeAnimation
                sequence={[caption]}
                wrapper="div"
                speed={50}
                style={{
                  fontSize: "1.25rem",
                  lineHeight: "1.75rem",
                  color: "#e5e7eb",
                  fontFamily: "Inter, sans-serif",
                }}
                cursor={false}
              />
            ) : (
              <div className="flex items-center space-x-3 text-gray-500">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 opacity-80" />
                  <div className="w-2 h-2 rounded-full bg-purple-400 opacity-80" />
                  <div className="w-2 h-2 rounded-full bg-pink-400 opacity-80" />
                </div>
                <span>Upload an image and click "Generate Caption"</span>
              </div>
            )}
          </div>
          <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-20"></div>
        </motion.div>
      </div>
    </motion.div>
  );
}

