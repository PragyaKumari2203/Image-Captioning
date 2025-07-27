import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './Components/Navbar';
import Footer from './components/Footer';
import Home from './Components/Home';
import About from './Components/About';
import NotFound from './Components/NotFound';
import CaptionBox from './Components/CaptionBox';

function Layout({ children }) {
  const location = useLocation();
  const isNotFoundPage = location.pathname === '*';

  return (
    <div className="flex flex-col min-h-screen">
      {!isNotFoundPage && <Navbar />}
      <main className="flex-grow">
        {children}
      </main>
      {!isNotFoundPage && <Footer />}
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="*" element={<NotFound />} />
        <Route path="/" element={
          <Layout>
            <Home />
          </Layout>
        } />
        <Route path="/caption" element={
          <Layout>
            <CaptionBox />
          </Layout>
        } />
        <Route path="/about" element={
          <Layout>
            <About />
          </Layout>
        } />
      </Routes>
    </Router>
  );
}