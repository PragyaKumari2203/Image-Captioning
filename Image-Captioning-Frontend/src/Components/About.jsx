import { motion } from "framer-motion";
import { FiCpu, FiServer, FiCode } from "react-icons/fi";

export default function About() {
  const techStack = [
    {
      title: "Machine Learning",
      icon: <FiCpu className="text-3xl" />,
      color: "from-indigo-900 to-indigo-950",
      features: [
        "PyTorch for core tensor operations & training",
        "ResNet101 + LSTM with Attention",
        "Custom tokenization & vocabulary",
        "Beam Search for caption generation",
        "NLTK for evaluation (BLEU-4)"
      ],
      details: [
        "Image preprocessing with TorchVision (resize, normalize)",
        "Special tokens: <start>, <pad>, <unk>, <end>",
        "PIL for image loading and augmentations",
        "GPU-accelerated training (CUDA via PyTorch)",
        "Model size: ~200MB"
      ]
    },
    {
      title: "Backend",
      icon: <FiServer className="text-3xl" />,
      color: "from-rose-900 to-rose-950",
      features: [
        "CORS for model interaction",
        "Model accessible through Google Drive",
        "Efficient image handling",
        "Scalable architecture",
        "Secure endpoints"
      ],
      details: [
        "Optimized for concurrent requests",
        "Handles model loading/unloading",
        "Manages preprocessing pipeline",
        "Error handling for large payloads",
        "Caching for frequent requests"
      ]
    },
    {
      title: "Frontend",
      icon: <FiCode className="text-3xl" />,
      color: "from-emerald-900 to-emerald-950",
      features: [
        "React.js for dynamic UI",
        "Tailwind CSS for styling",
        "Framer Motion animations",
        "Responsive design",
        "Interactive image upload"
      ],
      details: [
        "Smooth loading transitions",
        "Typewriter effect for captions",
        "Drag & drop image support",
        "Progressive loading states",
        "Accessibility optimized"
      ]
    }
  ];

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="relative min-h-screen p-8 bg-gradient-to-br from-gray-950 via-slate-950 to-gray-900 overflow-hidden"
    >
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-indigo-900/20 blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute bottom-1/3 right-1/4 w-72 h-72 rounded-full bg-rose-900/15 blur-3xl"
          animate={{
            scale: [1, 1.1, 1],
            opacity: [0.2, 0.4, 0.2]
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2
          }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-6xl mx-auto py-12">
        {/* Header */}
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, type: "spring" }}
          className="text-center mb-12 pt-8" 
        >
          <motion.h2
            className="text-4xl md:text-5xl font-bold mb-6 leading-tight" 
            animate={{
              backgroundPosition: ["0% 50%", "100% 50%"]
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              repeatType: "mirror",
              ease: "linear"
            }}
            style={{
              background: "linear-gradient(90deg, #6366f1, #ec4899, #6366f1)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
              backgroundSize: "200% auto"
            }}
          >
            About Our Project
          </motion.h2>
        </motion.div>

        {/* Tech stack grid - Equal height boxes */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20">
          {techStack.map((tech, i) => (
            <motion.div
              key={i}
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 * i, duration: 0.6 }}
              whileHover={{ y: -5 }}
              className={`flex flex-col bg-gradient-to-br ${tech.color} rounded-2xl shadow-2xl border border-white/10 overflow-hidden h-full`}
            >
              {/* Header section */}
              <div className="p-6 text-center border-b border-white/10">
                <motion.div 
                  className="flex justify-center mb-4 text-white"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 1 }}
                >
                  {tech.icon}
                </motion.div>
                <h3 className="text-xl font-bold text-white">{tech.title}</h3>
              </div>
              
              {/* Content section - made equal height */}
              <div className="flex-grow p-6">
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-gray-200 mb-3">KEY FEATURES</h4>
                  <ul className="space-y-2">
                    {tech.features.map((feature, j) => (
                      <li key={j} className="text-gray-300 text-sm flex items-start">
                        <span className="text-indigo-300 mr-2">•</span> 
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-black/20 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-200 mb-2">TECH DETAILS</h4>
                  <ul className="space-y-1">
                    {tech.details.map((detail, k) => (
                      <li key={k} className="text-gray-300 text-xs">
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* System overview */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
          className="bg-gray-900/70 backdrop-blur-lg border border-white/10 rounded-2xl p-8 max-w-4xl mx-auto shadow-xl relative overflow-hidden"
        >
          <h3 className="text-2xl font-bold text-center mb-6">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-rose-400">
              System Workflow
            </span>
          </h3>
          
          <div className="text-gray-300 space-y-4">
            {[
              "Frontend captures user image and sends to backend via CORS request",
              "Backend preprocesses image and sends to ML model hosted on Google Drive",
              "Resnet101 extracts image features which are fed into LSTM with attention",
              "Model generates caption using beam search and special tokens",
              "Caption returns through backend to frontend for display with animations"
            ].map((step, i) => (
              <p key={i} className="flex items-start">
                <span className="text-indigo-300 mr-2">{i + 1}.</span> 
                <span>{step}</span>
              </p>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Floating particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full bg-white/10"
          style={{
            width: `${Math.random() * 6 + 2}px`,
            height: `${Math.random() * 6 + 2}px`,
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
          }}
          animate={{
            y: [0, Math.random() * 100 - 50],
            x: [0, Math.random() * 50 - 25],
            opacity: [0.2, 0.8, 0.2]
          }}
          transition={{
            duration: Math.random() * 10 + 5,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      ))}
    </motion.div>
  );
}