import { useState, useEffect } from 'react';
import { MultiscaleContextModel } from '../utils/mcm';
import { predictHalfLife } from '../services/api';
import Navbar from '../components/Navbar';

const DUOLINGO_COLORS = {
  primary: '#58CC02',
  blue: '#1CB0F6',
  orange: '#FF9600',
  purple: '#CE82FF',
  red: '#FF4B4B',
  yellow: '#FFC800',
  gray: '#4B4B4B',
  lightGray: '#777777',
  border: '#E5E5E5',
  bg: '#F7F7F7',
  white: 'white'
};

const DEFAULT_FEATURES = [
  { name: 'Mcm Baseline', value: 0.45 },
  { name: 'Historical Accuracy', value: 0.32 },
  { name: 'Time Lag', value: -0.28 },
  { name: 'Part Of Speech', value: -0.15 },
  { name: 'Tense', value: -0.12 },
  { name: 'Hour Of Day', value: 0.08 },
  { name: 'Times Correct', value: 0.06 },
  { name: 'Times Wrong', value: -0.05 }
];



function getRecallStatus(pPred) {
  if (pPred >= 0.8) return { color: DUOLINGO_COLORS.primary, bg: DUOLINGO_COLORS.bg, border: DUOLINGO_COLORS.primary, label: 'strong' };
  if (pPred >= 0.5) return { color: DUOLINGO_COLORS.yellow, bg: DUOLINGO_COLORS.bg, border: DUOLINGO_COLORS.yellow, label: 'fading' };
  return { color: DUOLINGO_COLORS.red, bg: DUOLINGO_COLORS.bg, border: DUOLINGO_COLORS.red, label: 'weak' };
}

function formatFeatureName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export default function MemoryEngine() {
  const [lang, setLang] = useState('en->es');
  const [posLabel, setPosLabel] = useState('noun');
  const [tense, setTense] = useState('present_indicative');
  const [person, setPerson] = useState('1st_person');
  const [gramNumber, setGramNumber] = useState('singular');
  const [gender, setGender] = useState('masculine');
  const [caseVal, setCaseVal] = useState('nominative');
  const [definiteness, setDefiniteness] = useState('definite');
  const [degree, setDegree] = useState('comparative');
  const [historySeen, setHistorySeen] = useState(15);
  const [historyCorrect, setHistoryCorrect] = useState(10);
  const [timeLagDays, setTimeLagDays] = useState(7);
  const [hourOfDay, setHourOfDay] = useState(14);
  const [historyMode, setHistoryMode] = useState('fixed');
  const [compareMode, setCompareMode] = useState(false);
  const [mcmP, setMcmP] = useState(0.7);
  const [hPreds, setHPreds] = useState({ xgboost: 10.0, pimsleur: 10.0, leitner: 10.0, hlr: 10.0 });
  const [pPred, setPPred] = useState(0.7);
  const [shapFeatures, setShapFeatures] = useState([]);
  const [predictedTimeLag, setPredictedTimeLag] = useState(7);
  const [displayedAccuracy, setDisplayedAccuracy] = useState(0.667);

  const handlePredict = async () => {
    const mcm = new MultiscaleContextModel();
    if (historyMode === 'fixed') {
      for (let i = 0; i < historySeen; i++) mcm.study(0.5, i < historyCorrect);
    } else {
      [1.0, 2.0, 4.0, 7.0].slice(0, historySeen).forEach((interval, i) => mcm.study(interval, i < historyCorrect));
    }
    const mcmP = mcm.predict(timeLagDays);
    const acc = historyCorrect / Math.max(historySeen, 1);
    setDisplayedAccuracy(acc);
    const h = await predictHalfLife({
      hour_of_day: hourOfDay,
      time_lag_days: timeLagDays,
      lang,
      mcm_predicted_p: mcmP,
      historical_accuracy: acc,
      history_seen: historySeen,
      history_correct: historyCorrect,
      pos_label: posLabel,
      tense,
      person,
      grammatical_number: gramNumber,
      gender,
      case: caseVal,
      definiteness,
      degree
    });
    const p = Math.pow(2, -timeLagDays / h.xgboost);
    setMcmP(mcmP);
    setHPreds(h);
    setPPred(p);
    setShapFeatures(h.shap_values || []);
    setPredictedTimeLag(timeLagDays);
  };

  const updateChart = () => {
    const maxDays = Math.max(30, hPreds.xgboost * 2.5);
    const days = Array.from({ length: 100 }, (_, i) => (i / 99) * maxDays);
    const calcRecall = (d, h) => Math.pow(2, -d / h);
    
    const traces = [
      { x: days, y: days.map(d => calcRecall(d, hPreds.xgboost)), type: 'scatter', mode: 'lines', name: 'XGBoost', line: { color: DUOLINGO_COLORS.blue, width: 3 }, fill: 'tozeroy', fillcolor: 'rgba(28,176,246,0.08)' },
      { x: days, y: days.map(d => calcRecall(d, hPreds.pimsleur)), type: 'scatter', mode: 'lines', name: 'Pimsleur', line: { color: DUOLINGO_COLORS.orange, width: 2 } },
      { x: days, y: days.map(d => calcRecall(d, hPreds.leitner)), type: 'scatter', mode: 'lines', name: 'Leitner', line: { color: DUOLINGO_COLORS.purple, width: 2 } },
      { x: days, y: days.map(d => calcRecall(d, hPreds.hlr)), type: 'scatter', mode: 'lines', name: 'HLR', line: { color: DUOLINGO_COLORS.red, width: 2 } }
    ];
    
    if (compareMode) {
      traces.push({ x: days, y: days.map(d => calcRecall(d, hPreds.xgboost * 0.5)), type: 'scatter', mode: 'lines', name: 'Fast Forgetter', line: { color: DUOLINGO_COLORS.primary, width: 2, dash: 'dash' } });
    }
    
    traces.push({ x: [timeLagDays], y: [pPred], type: 'scatter', mode: 'markers', name: 'Today', marker: { color: '#000000', size: 14, symbol: 'diamond', line: { width: 2, color: DUOLINGO_COLORS.white } } });
    
    window.Plotly.newPlot('chart', traces, {
      xaxis: { title: 'Days Since Last Practice', gridcolor: '#eee' },
      yaxis: { title: 'Recall Probability', range: [0, 1.05], tickformat: '.0%', gridcolor: '#eee' },
      plot_bgcolor: DUOLINGO_COLORS.white,
      paper_bgcolor: DUOLINGO_COLORS.white,
      hovermode: 'x unified',
      legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
      shapes: [{ type: 'line', x0: 0, x1: maxDays, y0: 0.5, y1: 0.5, line: { color: DUOLINGO_COLORS.red, width: 2, dash: 'dot' } }],
      margin: { t: 40, b: 60, l: 60, r: 40 }
    }, { responsive: true });
  };

  useEffect(() => { handlePredict(); }, []);
  useEffect(() => { if (hPreds.xgboost !== 10.0) updateChart(); }, [hPreds, pPred, timeLagDays, compareMode]);

  const features = shapFeatures.length > 0
    ? shapFeatures.map(([name, value]) => ({ name: formatFeatureName(name), value }))
    : DEFAULT_FEATURES;

  const maxAbsValue = Math.max(...features.map(f => Math.abs(f.value)), 0.01);
  const daysToThreshold = pPred >= 0.5 ? hPreds.xgboost * Math.log2(2.0) : 0;
  const remaining = Math.max(0, daysToThreshold - predictedTimeLag);
  const recallStatus = getRecallStatus(pPred);

  return (
    <div className="min-h-screen max-w-7xl mx-auto p-8" style={{ fontFamily: 'Nunito, sans-serif' }}>
      <Navbar />

      <div className="rounded-2xl shadow-lg p-6 mb-6" style={{ backgroundColor: '#F7F7F7', border: '2px solid #E5E5E5' }}>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Language Track</label>
            <select value={lang} onChange={e => setLang(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="en->es">English → Spanish</option>
              <option value="en->fr">English → French</option>
              <option value="en->de">English → German</option>
              <option value="en->it">English → Italian</option>
              <option value="en->pt">English → Portuguese</option>
              <option value="es->en">Spanish → English</option>
              <option value="it->en">Italian → English</option>
              <option value="pt->en">Portuguese → English</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Part of Speech</label>
            <select value={posLabel} onChange={e => setPosLabel(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="noun">Noun</option>
              <option value="verb_lexical">Verb (lexical)</option>
              <option value="verb_ser">Verb (ser/estar)</option>
              <option value="verb_auxiliary">Verb (auxiliary)</option>
              <option value="verb_modal">Verb (modal)</option>
              <option value="adjective">Adjective</option>
              <option value="adverb">Adverb</option>
              <option value="determiner">Determiner</option>
              <option value="pronoun">Pronoun</option>
              <option value="preposition">Preposition</option>
              <option value="conjunction">Conjunction</option>
              <option value="proper_noun">Proper Noun</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Tense / Mood</label>
            <select value={tense} onChange={e => setTense(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="present_indicative">Present Indicative</option>
              <option value="past_participle">Past Participle</option>
              <option value="infinitive">Infinitive</option>
              <option value="gerund">Gerund</option>
              <option value="preterite">Preterite</option>
              <option value="conditional">Conditional</option>
              <option value="future_indicative">Future Indicative</option>
              <option value="imperative">Imperative</option>
              <option value="past_imperfect_indicative">Past Imperfect Indicative</option>
              <option value="present_subjunctive">Present Subjunctive</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Days Since Last Practice: <span style={{ color: '#58CC02' }}>{timeLagDays.toFixed(1)}</span></label>
            <input type="range" min="0.1" max="365" step="0.5" value={timeLagDays} onChange={e => setTimeLagDays(parseFloat(e.target.value))} className="w-full h-3 rounded-lg appearance-none cursor-pointer" style={{ background: '#E5E5E5', accentColor: '#58CC02' }} />
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Person</label>
            <select value={person} onChange={e => setPerson(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="1st_person">1st Person</option>
              <option value="2nd_person">2nd Person</option>
              <option value="3rd_person">3rd Person</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Number</label>
            <select value={gramNumber} onChange={e => setGramNumber(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="singular">Singular</option>
              <option value="plural">Plural</option>
              <option value="singular_or_plural">Singular or Plural</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Gender</label>
            <select value={gender} onChange={e => setGender(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="masculine">Masculine</option>
              <option value="feminine">Feminine</option>
              <option value="neuter">Neuter</option>
              <option value="masculine_or_feminine">Masculine or Feminine</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Hour of Day: <span style={{ color: '#58CC02' }}>{hourOfDay.toFixed(1)}</span></label>
            <input type="range" min="0" max="10" step="0.1" value={hourOfDay} onChange={e => setHourOfDay(parseFloat(e.target.value))} className="w-full h-3 rounded-lg appearance-none cursor-pointer" style={{ background: '#E5E5E5', accentColor: '#58CC02' }} />
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Case</label>
            <select value={caseVal} onChange={e => setCaseVal(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="nominative">Nominative</option>
              <option value="accusative">Accusative</option>
              <option value="dative">Dative</option>
              <option value="genitive">Genitive</option>
              <option value="vocative">Vocative</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Definiteness</label>
            <select value={definiteness} onChange={e => setDefiniteness(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="definite">Definite</option>
              <option value="indefinite">Indefinite</option>
              <option value="demonstrative">Demonstrative</option>
              <option value="possessive">Possessive</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Adj. Degree</label>
            <select value={degree} onChange={e => setDegree(e.target.value)} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }}>
              <option value="comparative">Comparative</option>
              <option value="superlative">Superlative</option>
            </select>
          </div>
          <div className="flex items-center justify-center gap-3">
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="radio" checked={historyMode === 'fixed'} onChange={() => setHistoryMode('fixed')} style={{ accentColor: '#58CC02' }} />
              <span className="text-sm font-semibold" style={{ color: '#4B4B4B' }}>Fixed</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="radio" checked={historyMode === 'spaced'} onChange={() => setHistoryMode('spaced')} style={{ accentColor: '#58CC02' }} />
              <span className="text-sm font-semibold" style={{ color: '#4B4B4B' }}>Spaced</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={compareMode} onChange={e => setCompareMode(e.target.checked)} style={{ accentColor: '#58CC02' }} />
              <span className="text-sm font-semibold" style={{ color: '#4B4B4B' }}>Compare</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Times Practiced</label>
            <input type="number" min="1" max="200" value={historySeen} onChange={e => { const val = parseInt(e.target.value) || 1; setHistorySeen(val); if (historyCorrect > val) setHistoryCorrect(val); }} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }} />
          </div>
          <div>
            <label className="block text-sm font-semibold mb-1" style={{ color: '#4B4B4B' }}>Times Correct</label>
            <input type="number" min="0" max={historySeen} value={historyCorrect} onChange={e => setHistoryCorrect(Math.min(parseInt(e.target.value) || 0, historySeen))} className="w-full border-2 rounded-xl px-3 py-2 focus:outline-none" style={{ borderColor: '#E5E5E5', backgroundColor: 'white' }} />
          </div>
          <div className="flex items-center justify-center col-span-2">
            <button onClick={handlePredict} className="font-bold px-12 py-3 rounded-2xl transition shadow-lg" style={{ backgroundColor: '#58CC02', color: 'white', fontSize: '18px' }}>Predict Memory</button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: recallStatus.bg, border: `3px solid ${recallStatus.border}` }}>
          <div className="text-5xl font-extrabold" style={{ color: recallStatus.color }}>{(pPred * 100).toFixed(1)}%</div>
          <div className="text-sm font-semibold mt-2" style={{ color: '#777777' }}>recall — <b>{recallStatus.label}</b></div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>Accuracy</div>
          <div className="text-3xl font-extrabold" style={{ color: '#4B4B4B' }}>{(displayedAccuracy * 100).toFixed(1)}%</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>Review In</div>
          <div className="text-2xl font-extrabold" style={{ color: pPred >= 0.5 ? '#1CB0F6' : '#FF4B4B' }}>{pPred >= 0.5 ? `${remaining.toFixed(1)} Days` : 'Now'}</div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>XGBoost</div>
          <div className="text-3xl font-extrabold" style={{ color: '#1CB0F6' }}>{hPreds.xgboost.toFixed(1)} Days</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>Pimsleur</div>
          <div className="text-3xl font-extrabold" style={{ color: '#FF9600' }}>{hPreds.pimsleur.toFixed(1)} Days</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>Leitner</div>
          <div className="text-3xl font-extrabold" style={{ color: '#CE82FF' }}>{hPreds.leitner.toFixed(1)} Days</div>
        </div>
        <div className="rounded-2xl p-6 text-center shadow-lg" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div className="text-xs font-bold uppercase mb-2" style={{ color: '#777777' }}>HLR</div>
          <div className="text-3xl font-extrabold" style={{ color: '#FF4B4B' }}>{hPreds.hlr.toFixed(1)} Days</div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="rounded-2xl shadow-lg p-6" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          <div id="chart" style={{ width: '100%', height: '400px' }}></div>
        </div>
        <div className="rounded-2xl shadow-lg p-6" style={{ backgroundColor: 'white', border: '2px solid #E5E5E5' }}>
          {features.map(f => (
            <div key={f.name} className="flex items-center mb-3">
              <div className="w-40 text-sm font-semibold" style={{ color: '#4B4B4B' }}>{f.name}</div>
              <div className="flex-1 h-7 rounded-lg relative overflow-hidden" style={{ backgroundColor: '#F7F7F7' }}>
                <div className="h-full" style={{ width: `${(Math.abs(f.value) / maxAbsValue) * 100}%`, backgroundColor: f.value > 0 ? '#58CC02' : '#FF4B4B' }}></div>
              </div>
              <div className="w-20 text-right text-sm font-bold" style={{ color: f.value > 0 ? '#58CC02' : '#FF4B4B' }}>{f.value > 0 ? '+' : ''}{f.value.toFixed(1)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
