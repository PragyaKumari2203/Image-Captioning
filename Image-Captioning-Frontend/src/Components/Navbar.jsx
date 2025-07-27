import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

import logo from '../assets/logo6.png'; 

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${scrolled ? 'py-0' : 'py-1'}`}>
      <nav className={`bg-gradient-to-b from-black/90 to-black/70 backdrop-blur-lg border-b ${scrolled ? 'border-white/10' : 'border-transparent'} shadow-xl px-2 py-1 md:px-4 md:py-2 flex justify-between items-center max-w-full md:max-w-7xl mx-auto rounded-b-2xl transition-all duration-500`}>
        <Link to="/" className="flex items-center space-x-2 group">
        <img
  src={logo}
  alt="Logo"
  className="h-16 w-20 object-contain transition-transform duration-300 group-hover:scale-105"
/>

</Link>


        {/* Desktop Nav */}
        <div className="hidden md:flex items-center space-x-8">
          <NavItem to="/" label="Home" currentPath={location.pathname} />
          <NavItem to="/caption" label="Caption" currentPath={location.pathname} />
          <NavItem to="/about" label="About" currentPath={location.pathname} />
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`md:hidden p-2 rounded-full transition-all ${isOpen ? 'bg-white/10' : 'hover:bg-white/10'}`}
          aria-label="Toggle Menu"
        >
          {isOpen ? (
            <X size={28} className="text-purple-400" />
          ) : (
            <Menu size={28} className="text-white" />
          )}
        </button>
      </nav>

      {/* Mobile Nav Items */}
      <div
        className={`md:hidden transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] bg-gradient-to-b from-black/95 to-black/90 backdrop-blur-xl shadow-2xl overflow-hidden ${
          isOpen ? 'max-h-60 py-4 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="flex flex-col items-center space-y-5 px-4">
          <NavItem 
            to="/" 
            label="Home" 
            onClick={() => setIsOpen(false)} 
            currentPath={location.pathname} 
            mobile
          />
          <NavItem 
            to="/caption" 
            label="Caption" 
            onClick={() => setIsOpen(false)} 
            currentPath={location.pathname} 
            mobile
          />
          <NavItem 
            to="/about" 
            label="About" 
            onClick={() => setIsOpen(false)} 
            currentPath={location.pathname} 
            mobile
          />
        </div>
      </div>
    </header>
  );
}

function NavItem({ to, label, onClick, currentPath, mobile = false }) {
  const isActive = currentPath === to;

  return (
    <Link
      to={to}
      onClick={onClick}
      className={`relative group text-lg font-medium tracking-wide transition-all duration-300 px-3 py-1 ${
        isActive ? 'text-purple-400' : 'text-gray-300 hover:text-purple-300'
      } ${mobile ? 'w-full text-center' : ''}`}
    >
      <span className="relative z-10">
        {label}
        {isActive && (
          <span className="absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full" />
        )}
      </span>
      {!isActive && (
        <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full transition-all duration-300 group-hover:w-full"></span>
      )}
    </Link>
  );
}