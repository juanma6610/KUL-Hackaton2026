import { useState, useEffect } from 'react';
import { fetchUserStats } from '../services/api';
import Navbar from '../components/Navbar';

const DUOLINGO_COLORS = {
  primary: '#58CC02',
  blue: '#1CB0F6',
  orange: '#FF9600',
  gray: '#4B4B4B',
  lightGray: '#777777',
  border: '#E5E5E5',
  bg: '#F7F7F7',
  white: 'white',
  red: '#FF4B4B'
};

const FALLBACK_STATS = {
  total_words: 450,
  accuracy: 0.85,
  streak_days: 25,
  weak_categories: [
    { name: 'Verbs (Past)', count: 23 },
    { name: 'Adjectives', count: 18 },
    { name: 'Pronouns', count: 15 }
  ],
  time_distribution: Array.from({ length: 24 }, (_, i) => ({ hour: i, count: Math.floor(Math.random() * 50) })),
  accuracy_trend: Array.from({ length: 30 }, (_, i) => ({ day: i, accuracy: 0.6 + Math.random() * 0.3 }))
};

function createPlot(elementId, data, layout) {
  window.Plotly.newPlot(elementId, data, {
    ...layout,
    plot_bgcolor: DUOLINGO_COLORS.white,
    paper_bgcolor: DUOLINGO_COLORS.white,
    margin: { t: 20, b: 40, l: 50, r: 20 }
  }, { responsive: true });
}

export default function UserProfile() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchUserStats().then(data => setStats(data || FALLBACK_STATS));
  }, []);

  useEffect(() => {
    if (!stats) return;

    createPlot('timeChart', [{
      x: stats.time_distribution.map(d => d.hour),
      y: stats.time_distribution.map(d => d.count),
      type: 'bar',
      marker: { color: DUOLINGO_COLORS.blue }
    }], {
      xaxis: { title: 'Hour of Day' },
      yaxis: { title: 'Practice Sessions' }
    });

    createPlot('accuracyChart', [{
      x: stats.accuracy_trend.map(d => d.day),
      y: stats.accuracy_trend.map(d => d.accuracy),
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: DUOLINGO_COLORS.primary, width: 3 },
      marker: { size: 6 }
    }], {
      xaxis: { title: 'Days Ago' },
      yaxis: { title: 'Accuracy', tickformat: '.0%' }
    });
  }, [stats]);

  if (!stats) return <div className="text-center p-8">Loading...</div>;

  return (
    <div className="min-h-screen max-w-7xl mx-auto p-8" style={{ fontFamily: 'Nunito, sans-serif' }}>
      <Navbar />

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: DUOLINGO_COLORS.lightGray }}>Total Words</div>
          <div className="text-4xl font-extrabold" style={{ color: DUOLINGO_COLORS.blue }}>{stats.total_words}</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: DUOLINGO_COLORS.lightGray }}>Overall Accuracy</div>
          <div className="text-4xl font-extrabold" style={{ color: DUOLINGO_COLORS.primary }}>{(stats.accuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: DUOLINGO_COLORS.lightGray }}>Streak</div>
          <div className="text-4xl font-extrabold" style={{ color: DUOLINGO_COLORS.orange }}>{stats.streak_days} Days</div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="rounded-2xl shadow-lg p-6" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
          <h3 className="text-xl font-bold mb-4" style={{ color: DUOLINGO_COLORS.gray }}>Practice Time Distribution</h3>
          <div id="timeChart" style={{ width: '100%', height: '300px' }}></div>
        </div>
        <div className="rounded-2xl shadow-lg p-6" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
          <h3 className="text-xl font-bold mb-4" style={{ color: DUOLINGO_COLORS.gray }}>Accuracy Trend (30 Days)</h3>
          <div id="accuracyChart" style={{ width: '100%', height: '300px' }}></div>
        </div>
      </div>

      <div className="rounded-2xl shadow-lg p-6" style={{ backgroundColor: DUOLINGO_COLORS.white, border: `2px solid ${DUOLINGO_COLORS.border}` }}>
        <h3 className="text-xl font-bold mb-4" style={{ color: DUOLINGO_COLORS.gray }}>Weak Categories - Focus Here</h3>
        {stats.weak_categories.map((cat, i) => (
          <div key={i} className="flex items-center justify-between mb-3 p-3 rounded-xl" style={{ backgroundColor: DUOLINGO_COLORS.bg }}>
            <span className="font-semibold" style={{ color: DUOLINGO_COLORS.gray }}>{cat.name}</span>
            <span className="font-bold" style={{ color: DUOLINGO_COLORS.red }}>{cat.count} words need review</span>
          </div>
        ))}
      </div>
    </div>
  );
}
