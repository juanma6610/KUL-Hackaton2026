import { Link, useLocation } from 'react-router-dom';

const PAGES = [
  { path: '/', name: 'Memory Engine' },
  { path: '/profile', name: 'User Profile' }
];

export default function Navbar() {
  const location = useLocation();
  const currentPage = PAGES.find(p => p.path === location.pathname)?.name || 'Memory Engine';

  return (
    <div className="flex items-center justify-between mb-8 p-4 rounded-2xl" style={{ backgroundColor: '#F7F7F7', border: '2px solid #E5E5E5' }}>
      <img src="https://design.duolingo.com/86230c9ad10d9f08b785.svg" alt="Duo" className="w-12 h-12" />
      <h1 className="text-3xl font-extrabold" style={{ color: '#58CC02' }}>{currentPage}</h1>
      <div className="flex gap-3">
        {PAGES.map(page => (
          <Link
            key={page.path}
            to={page.path}
            className="px-4 py-2 rounded-xl font-semibold text-sm transition"
            style={{
              backgroundColor: location.pathname === page.path ? '#58CC02' : '#E5E5E5',
              color: location.pathname === page.path ? 'white' : '#4B4B4B'
            }}
          >
            {page.name}
          </Link>
        ))}
      </div>
    </div>
  );
}
