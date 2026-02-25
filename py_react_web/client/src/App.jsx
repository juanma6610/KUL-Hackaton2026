import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MemoryEngine from './pages/MemoryEngine';
import UserProfile from './pages/UserProfile';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MemoryEngine />} />
        <Route path="/profile" element={<UserProfile />} />
      </Routes>
    </BrowserRouter>
  );
}
