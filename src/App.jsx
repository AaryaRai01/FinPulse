import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Activity, BookOpen, BrainCircuit, ChevronRight, CircleHelp, Database,
  FileSearch, LayoutDashboard,
  Menu, Search, ShieldCheck, Workflow, X
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie,
  PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

const COLORS = ["#3976b9", "#d18b38", "#2f9a83"];
const NAV = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "case-study", label: "Case study", icon: BookOpen },
  { id: "model", label: "ML training & implementation", icon: BrainCircuit },
  { id: "explorer", label: "Results explorer", icon: FileSearch }
];

function App() {
  const [page, setPage] = useState("overview");
  const [summary, setSummary] = useState(null);
  const [records, setRecords] = useState([]);
  const [dataStatus, setDataStatus] = useState("loading");
  const [query, setQuery] = useState("");
  const [yearFilter, setYearFilter] = useState("all");
  const [stateFilter, setStateFilter] = useState("all");
  const [mobileNav, setMobileNav] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("/data/model_summary.json").then(r => {
        if (!r.ok) throw new Error("Could not load model summary");
        return r.json();
      }),
      fetch("/data/household_hidden_states.csv").then(r => {
        if (!r.ok) throw new Error("Could not load household results");
        return r.text();
      })
    ]).then(([s, csv]) => {
      setSummary(s);
      const parsed = Papa.parse(csv, { header: true, skipEmptyLines: true, dynamicTyping: true });
      setRecords(parsed.data);
      setDataStatus("ready");
    }).catch(err => {
      console.error(err);
      setDataStatus("error");
    });
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return records.filter(row => {
      const matchQuery = !q || String(row.HH_ID).toLowerCase().includes(q);
      const matchYear = yearFilter === "all" || String(row.year) === yearFilter;
      const matchState = stateFilter === "all" || String(row.hidden_state) === stateFilter;
      return matchQuery && matchYear && matchState;
    }).slice(0, 100);
  }, [records, query, yearFilter, stateFilter]);

  const activePage = NAV.find(n => n.id === page);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Activity size={21} /></div>
          <div><strong>FinPulse India</strong><span>ML case study dashboard</span></div>
          <button className="mobile-close" onClick={() => setMobileNav(false)}><X size={18}/></button>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav>
          {NAV.map(item => {
            const Icon = item.icon;
            return <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => { setPage(item.id); setMobileNav(false); }}>
              <Icon size={18}/><span>{item.label}</span>{page === item.id && <ChevronRight size={16} className="nav-arrow"/>}
            </button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="status-dot"></div>
          <div><strong>Model results loaded</strong><span>Static research dataset</span></div>
        </div>
      </aside>

      {mobileNav && <div className="scrim" onClick={() => setMobileNav(false)} />}
      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNav(true)}><Menu size={20}/></button>
          <div className="breadcrumb">FinPulse India <span>/</span> {activePage?.label}</div>
          <div className="topbar-right"><span className="dataset-pill"><span/> CPHS · 2016–2019</span></div>
        </header>
        <div className="content">
          {dataStatus === "loading" && <div className="loading-panel">Loading trained model results…</div>}
          {dataStatus === "error" && <div className="loading-panel error">Could not load project data. Check that files in <code>public/data</code> are present.</div>}
          {dataStatus === "ready" && summary && <>
            {page === "overview" && <Overview summary={summary} records={records}/>}
            {page === "case-study" && <CaseStudy summary={summary}/>}
            {page === "model" && <ModelWorkflow summary={summary} records={records}/>}
            {page === "explorer" && <Explorer records={filtered} query={query} setQuery={setQuery}
              yearFilter={yearFilter} setYearFilter={setYearFilter}
              stateFilter={stateFilter} setStateFilter={setStateFilter}
              total={records.length}/>}
          </>}
          <footer>FinPulse India · Academic machine learning project · Results are exploratory, not financial advice.</footer>
        </div>
      </main>
    </div>
  );
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading">
    <div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
    {action && <div className="heading-action">{action}</div>}
  </div>;
}

function StatCard({ label, value, note, icon: Icon }) {
  return <div className="stat-card">
    <div className="stat-top"><span>{label}</span><div className="stat-icon"><Icon size={18}/></div></div>
    <strong>{value}</strong><small>{note}</small>
  </div>;
}

function Panel({ title, subtitle, children, className = "" }) {
  return <section className={`panel ${className}`}>
    <div className="panel-heading"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div></div>
    {children}
  </section>;
}

function Overview({ summary, records }) {
  const states = summary.state_distribution;
  const profileData = summary.state_profiles.map(s => ({
    state: `State ${s.stateId}`, MDPS: s.MDPS, VMDPS: s.VMDPS, "Dependency rate": s.dep_rate
  }));
  return <>
    <PageHeading eyebrow="PROJECT OVERVIEW" title="Household vulnerability, viewed over time."
      description="An academic demonstration of how a Hidden Markov Model can identify latent household vulnerability profiles from annual survey observations."
      action={<span className="model-badge"><span/> HMM inference complete</span>}/>
    <div className="stats-grid">
      <StatCard label="Eligible observations" value={summary.dataset.eligible_observations.toLocaleString("en-IN")} note="Household-year records used for inference" icon={Database}/>
      <StatCard label="Sequence segments" value={summary.dataset.sequence_segments.toLocaleString("en-IN")} note="Consecutive annual segments" icon={Workflow}/>
      <StatCard label="Hidden states" value={summary.training.hidden_states} note="Inferred vulnerability profiles" icon={BrainCircuit}/>
      <StatCard label="Survey period" value="2016–2019" note="Four annual survey waves" icon={Activity}/>
    </div>
    <div className="chart-grid two-col">
      <Panel title="Hidden state distribution" subtitle="How the eligible observations are assigned across the three inferred states.">
        <div className="chart-wrap"><ResponsiveContainer width="100%" height={270}>
          <BarChart data={states} margin={{top: 12, right: 12, left: -18, bottom: 4}}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
            <XAxis dataKey="state" tickLine={false} axisLine={false}/>
            <YAxis tickLine={false} axisLine={false}/>
            <Tooltip formatter={(v, n) => [Number(v).toLocaleString("en-IN"), n]}/>
            <Bar dataKey="observations" name="Observations" radius={[6,6,0,0]}>
              {states.map((s, i) => <Cell key={s.stateId} fill={COLORS[i]}/>)}
            </Bar>
          </BarChart>
        </ResponsiveContainer></div>
        <div className="legend-row">{states.map((s,i)=><div key={s.stateId}><i style={{background:COLORS[i]}}/>{s.state} <b>{s.percentage}%</b></div>)}</div>
      </Panel>
      <Panel title="Annual state composition" subtitle="Percentage of eligible observations in each state by survey year.">
        <div className="chart-wrap"><ResponsiveContainer width="100%" height={270}>
          <BarChart data={summary.annual_distribution} margin={{top: 12, right: 10, left: -18, bottom: 4}}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
            <XAxis dataKey="year" tickLine={false} axisLine={false}/>
            <YAxis domain={[0,100]} tickFormatter={v=>`${v}%`} tickLine={false} axisLine={false}/>
            <Tooltip formatter={v=>`${v}%`}/>
            <Legend/>
            <Bar dataKey="state0" name="State 0" stackId="a" fill={COLORS[0]}/>
            <Bar dataKey="state1" name="State 1" stackId="a" fill={COLORS[1]}/>
            <Bar dataKey="state2" name="State 2" stackId="a" fill={COLORS[2]} radius={[5,5,0,0]}/>
          </BarChart>
        </ResponsiveContainer></div>
      </Panel>
    </div>
    <div className="chart-grid two-col">
      <Panel title="Average feature values by state" subtitle="Original feature scale; higher MDPS and VMDPS indicate higher observed scores.">
        <div className="chart-wrap"><ResponsiveContainer width="100%" height={270}>
          <BarChart data={profileData} margin={{top: 12, right: 12, left: -18, bottom: 4}}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
            <XAxis dataKey="state" tickLine={false} axisLine={false}/>
            <YAxis domain={[0,0.7]} tickLine={false} axisLine={false}/>
            <Tooltip formatter={v=>Number(v).toFixed(3)}/>
            <Legend/>
            <Bar dataKey="MDPS" fill="#3976b9" radius={[4,4,0,0]}/>
            <Bar dataKey="VMDPS" fill="#d18b38" radius={[4,4,0,0]}/>
            <Bar dataKey="Dependency rate" fill="#2f9a83" radius={[4,4,0,0]}/>
          </BarChart>
        </ResponsiveContainer></div>
      </Panel>
      <Panel title="Learned transition probabilities" subtitle="Probability of moving from the current state (row) to the next annual state (column).">
        <TransitionMatrix matrix={summary.transition_matrix}/>
        <div className="small-note">Diagonal cells show the estimated probability of remaining in the same state.</div>
      </Panel>
    </div>
    <Panel title="Three inferred profiles" subtitle="Labels are descriptive summaries of the observed MDPS and VMDPS means, not official poverty categories.">
      <div className="state-cards">{summary.state_profiles.map((s,i)=><div className="state-card" key={s.stateId}>
        <div className="state-card-top"><span className="state-dot" style={{background:COLORS[s.stateId]}}/> State {s.stateId}</div>
        <h4>{s.label}</h4>
        <div className="state-metrics"><span>MDPS <b>{s.MDPS.toFixed(3)}</b></span><span>VMDPS <b>{s.VMDPS.toFixed(3)}</b></span></div>
        <small>{s.count.toLocaleString("en-IN")} observations</small>
      </div>)}</div>
    </Panel>
  </>;
}

function TransitionMatrix({ matrix }) {
  return <div className="matrix-wrap"><table className="matrix-table">
    <thead><tr><th>From / To</th>{[0,1,2].map(i=><th key={i}>State {i}</th>)}</tr></thead>
    <tbody>{matrix.map((row,i)=><tr key={i}><th>State {i}</th>{row.map((v,j)=><td key={j} style={{background:`rgba(47,154,131,${0.07+v*0.35})`}}>{(v*100).toFixed(1)}%</td>)}</tr>)}</tbody>
  </table></div>;
}

function CaseStudy({ summary }) {
  return <>
    <PageHeading eyebrow="CASE STUDY" title="Why study household vulnerability?"
      description="The project uses an India-focused household survey research dataset to explore how multidimensional poverty and vulnerability profiles can change across annual observations."/>
    <div className="case-hero">
      <div className="case-icon"><BookOpen size={25}/></div>
      <div><span className="eyebrow">SELECTED RESEARCH CONTEXT</span>
        <h2>Impact of Microcredit on Multidimensional Poverty and Vulnerability in India</h2>
        <p>Research dataset associated with Rasmita Maharana and Tara Shankar Shaw.</p>
      </div>
    </div>
    <div className="info-grid">
      <Panel title="Dataset overview"><dl className="detail-list">
        <div><dt>Source</dt><dd>{summary.dataset.name}</dd></div>
        <div><dt>Survey years</dt><dd>2016, 2017, 2018 and 2019</dd></div>
        <div><dt>Original analytical records</dt><dd>{summary.dataset.original_records.toLocaleString("en-IN")}</dd></div>
        <div><dt>Records used for HMM inference</dt><dd>{summary.dataset.eligible_observations.toLocaleString("en-IN")}</dd></div>
        <div><dt>Sequence segments</dt><dd>{summary.dataset.sequence_segments.toLocaleString("en-IN")}</dd></div>
      </dl></Panel>
      <Panel title="What the project investigates">
        <p className="body-copy">The project groups annual household observations into latent states using three observed indicators. It then estimates how likely a household observation is to remain in, or transition between, these inferred states in the next consecutive survey year.</p>
        <div className="feature-list">
          <div><b>MDPS</b><span>Multidimensional Poverty Score</span></div>
          <div><b>VMDPS</b><span>Vulnerability to Multidimensional Poverty Score</span></div>
          <div><b>Dependency rate</b><span>Household dependency indicator used in the analytical dataset</span></div>
        </div>
      </Panel>
    </div>
    <Panel title="Scope and limitations" subtitle="These boundaries are important when interpreting the results.">
      <ul className="limit-list">{summary.limitations.map((item,i)=><li key={i}>{item}</li>)}</ul>
    </Panel>
  </>;
}

function ModelWorkflow({ summary, records }) {
  const [selectedSequence, setSelectedSequence] = useState("");
  const sequences = useMemo(() => {
    const grouped = new Map();
    records.forEach(row => {
      const key = String(row.sequence_id || row.HH_ID);
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    });
    return Array.from(grouped.entries())
      .map(([id, rows]) => ({ id, rows: rows.sort((a,b) => Number(a.year)-Number(b.year)) }))
      .filter(item => item.rows.length >= 2)
      .sort((a,b) => b.rows.length-a.rows.length || String(a.id).localeCompare(String(b.id)))
      .slice(0, 250);
  }, [records]);
  const activeSequence = sequences.find(item => item.id === selectedSequence) || sequences[0];
  const sequenceRows = activeSequence?.rows || [];
  const training = summary.training;
  const stateProfiles = summary.state_profiles.map(item => ({
    state: `State ${item.stateId}`,
    MDPS: item.MDPS,
    VMDPS: item.VMDPS,
    "Dependency rate": item.dep_rate
  }));
  const trainingCoverage = [
    { label: "Training subset", observations: training.training_observations },
    { label: "All eligible inference data", observations: summary.dataset.eligible_observations }
  ];
  const steps = [
    {
      number: "01", title: "Build household sequences",
      detail: "Sort each household by survey year. Keep only consecutive annual observations within a sequence; start a new segment when a year is missing.",
      reason: "An HMM models temporal transitions. Connecting 2016 directly to 2018 would incorrectly treat a two-year gap as one annual transition."
    },
    {
      number: "02", title: "Prepare and scale features",
      detail: "Use MDPS, VMDPS and dep_rate as the three-dimensional observation vector. Standardize the features before fitting the Gaussian emission distributions.",
      reason: "Scaling prevents a feature's numeric range from dominating the Gaussian likelihood merely because of its units."
    },
    {
      number: "03", title: "Fit the Gaussian HMM",
      detail: "Fit three latent states with tied covariance. The Baum–Welch expectation-maximisation procedure estimates the initial-state, transition and Gaussian emission parameters.",
      reason: "The hidden state is not supplied as a label. The model discovers latent groups that explain the observed sequences probabilistically."
    },
    {
      number: "04", title: "Decode the hidden-state path",
      detail: "Apply Viterbi decoding separately across valid sequence segments to obtain the most likely sequence of hidden states.",
      reason: "This gives one coherent state path for each household segment, rather than classifying each year independently."
    },
    {
      number: "05", title: "Calculate posterior probabilities",
      detail: "Use the Forward–Backward procedure to calculate the posterior probability of each hidden state at every observation.",
      reason: "A probability distribution communicates uncertainty; the Viterbi state alone does not."
    },
    {
      number: "06", title: "Interpret and visualise",
      detail: "Summarise the original-scale feature values by state, inspect the learned transition matrix and compare annual state composition.",
      reason: "State numbers are arbitrary. Their descriptive meaning comes from the observed feature profiles, not from the numeric ID."
    }
  ];

  return <>
    <PageHeading eyebrow="MACHINE LEARNING IMPLEMENTATION" title="How FinPulse learns household vulnerability states"
      description="A step-by-step view of the actual data pipeline, Gaussian HMM training, inference algorithms and diagnostics used in the Colab experiment."
      action={<span className="model-badge"><span/> Notebook training completed</span>}/>

    <div className="training-banner">
      <div className="training-icon"><ShieldCheck size={23}/></div>
      <div><strong>Gaussian HMM training converged</strong><span>{training.iterations} EM iterations · {training.hidden_states} latent states · tied covariance</span></div>
      <span className="completed-tag">Converged</span>
    </div>

    <div className="stats-grid">
      <StatCard label="Training observations" value={training.training_observations.toLocaleString("en-IN")} note="Observations used to estimate parameters" icon={Database}/>
      <StatCard label="Training sequences" value={training.training_sequences.toLocaleString("en-IN")} note="Consecutive household segments sampled" icon={Workflow}/>
      <StatCard label="EM iterations" value={training.iterations} note="Iterations completed before convergence" icon={Activity}/>
      <StatCard label="Hidden states" value={training.hidden_states} note="Latent states inferred by the HMM" icon={BrainCircuit}/>
    </div>

    <Panel title="1. Data used for training versus inference" subtitle="The model was fitted on a sampled subset, then applied to all eligible observations.">
      <div className="training-split">
        <div className="split-copy">
          <div className="split-number">{training.training_observations.toLocaleString("en-IN")} <span>/ {summary.dataset.eligible_observations.toLocaleString("en-IN")}</span></div>
          <p>Training observations out of all eligible household-year records.</p>
          <div className="split-track"><div style={{width:`${100*training.training_observations/summary.dataset.eligible_observations}%`}}/></div>
          <div className="split-foot"><b>{(100*training.training_observations/summary.dataset.eligible_observations).toFixed(1)}%</b> of eligible observations used for parameter estimation</div>
        </div>
        <div className="split-facts">
          <div><span>Original analytical records</span><b>{summary.dataset.original_records.toLocaleString("en-IN")}</b></div>
          <div><span>Eligible observations</span><b>{summary.dataset.eligible_observations.toLocaleString("en-IN")}</b></div>
          <div><span>All consecutive sequence segments</span><b>{summary.dataset.sequence_segments.toLocaleString("en-IN")}</b></div>
          <div><span>Segments sampled for training</span><b>{training.training_sequences.toLocaleString("en-IN")}</b></div>
        </div>
      </div>
      <div className="note-box compact-note"><CircleHelp size={17}/><p><b>Why sample?</b> Training on a subset reduced computation time. The fitted parameters were subsequently used to infer states for the complete eligible dataset. Training and inference counts are therefore different.</p></div>
    </Panel>

    <Panel title="2. Actual implementation pipeline" subtitle="What happens to the data, why each operation is needed, and what it produces.">
      <div className="implementation-steps">
        {steps.map((step,i)=><div className="implementation-step" key={step.number}>
          <div className="implementation-index">{step.number}</div>
          <div className="implementation-main">
            <h4>{step.title}</h4>
            <p>{step.detail}</p>
            <div className="why-line"><b>WHY IT MATTERS</b><span>{step.reason}</span></div>
          </div>
          {i < steps.length-1 && <div className="implementation-connector"/>}
        </div>)}
      </div>
    </Panel>

    <div className="info-grid">
      <Panel title="3. What the HMM learns" subtitle="Three parameter groups describe the hidden-state process.">
        <div className="parameter-card"><span>Initial state probabilities · π</span><strong>πᵢ = P(Z₁ = i)</strong><p>Probability that a sequence begins in hidden state i.</p></div>
        <div className="parameter-card"><span>Transition matrix · A</span><strong>Aᵢⱼ = P(Zₜ = j | Zₜ₋₁ = i)</strong><p>Probability of moving from state i to state j in the next consecutive survey wave.</p></div>
        <div className="parameter-card"><span>Gaussian emission model</span><strong>Xₜ | Zₜ = k ∼ N(μₖ, Σ)</strong><p>Each hidden state has its own mean vector. Tied covariance means the covariance matrix Σ is shared across all states.</p></div>
      </Panel>
      <Panel title="4. Training with Baum–Welch / EM" subtitle="The model alternates between estimating hidden-state responsibilities and updating parameters.">
        <div className="em-cycle">
          <div className="em-step"><span className="em-tag">E-STEP</span><b>Expectation</b><p>Forward–Backward calculates expected state occupancy and expected state-to-state transitions for the observed sequences.</p></div>
          <div className="em-arrow">↓</div>
          <div className="em-step"><span className="em-tag">M-STEP</span><b>Maximisation</b><p>Update initial probabilities, transition probabilities, Gaussian means and the shared covariance to increase data likelihood.</p></div>
          <div className="em-loop">Repeat until convergence criterion is met</div>
        </div>
        <div className="convergence-result"><div><strong>{training.iterations}</strong><span>iterations</span></div><div><strong>Yes</strong><span>converged</span></div><div><strong>{training.covariance_type}</strong><span>covariance</span></div></div>
        <p className="small-note">The notebook output records the final iteration count and convergence status. An iteration-by-iteration likelihood history was not exported, so this dashboard does not fabricate a training-loss curve.</p>
      </Panel>
    </div>

    <Panel title="5. Training configuration" subtitle="Values recorded from the completed hmmlearn GaussianHMM run.">
      <div className="config-grid">
        <Config label="Implementation" value="hmmlearn GaussianHMM"/>
        <Config label="Hidden states" value={training.hidden_states}/>
        <Config label="Covariance type" value={training.covariance_type}/>
        <Config label="Training sequences" value={training.training_sequences.toLocaleString("en-IN")}/>
        <Config label="Training observations" value={training.training_observations.toLocaleString("en-IN")}/>
        <Config label="EM iterations" value={training.iterations}/>
        <Config label="Observation features" value="MDPS · VMDPS · dep_rate"/>
        <Config label="Random seed" value={training.random_state}/>
      </div>
      <div className="code-explainer">
        <div><b>Implementation outline</b><span>Conceptual code showing the main hmmlearn calls used in this workflow.</span></div>
        <pre><code>{`model = GaussianHMM(
    n_components=3,
    covariance_type="tied",
    random_state=42
)

model.fit(X_train, lengths=train_lengths)

states = model.predict(X_all, lengths=all_lengths)
posterior = model.predict_proba(X_all, lengths=all_lengths)`}</code></pre>
        <p className="small-note">X_train and X_all contain the scaled feature vectors. The lengths arrays preserve the household sequence boundaries during training and inference.</p>
      </div>
    </Panel>

    <div className="chart-grid two-col">
      <Panel title="6. Learned Gaussian emission profiles" subtitle="Mean feature values by inferred state, shown in original dataset units.">
        <div className="chart-wrap"><ResponsiveContainer width="100%" height={270}>
          <BarChart data={stateProfiles} margin={{top:12,right:12,left:-18,bottom:4}}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
            <XAxis dataKey="state" tickLine={false} axisLine={false}/>
            <YAxis domain={[0,0.65]} tickLine={false} axisLine={false}/>
            <Tooltip formatter={v=>Number(v).toFixed(3)}/>
            <Legend/>
            <Bar dataKey="MDPS" fill="#3976b9" radius={[4,4,0,0]}/>
            <Bar dataKey="VMDPS" fill="#d18b38" radius={[4,4,0,0]}/>
            <Bar dataKey="Dependency rate" fill="#2f9a83" radius={[4,4,0,0]}/>
          </BarChart>
        </ResponsiveContainer></div>
        <p className="small-note">State 1 has the highest mean MDPS and VMDPS in this fitted model; State 2 has the lowest. These are descriptive profiles, not externally validated poverty classes.</p>
      </Panel>
      <Panel title="7. Learned transition matrix" subtitle="Rows are the current state; columns are the next state.">
        <TransitionMatrix matrix={summary.transition_matrix}/>
        <p className="small-note">For example, State 2 has an estimated 93.0% probability of remaining in State 2 at the next consecutive annual observation. This is a model-estimated transition probability, not a causal effect.</p>
      </Panel>
    </div>

    <Panel title="8. Household-level inference: inspect the actual output" subtitle="Choose a household sequence to see its observed features, Viterbi state path and Forward–Backward posterior probabilities.">
      {sequences.length > 0 ? <>
        <div className="sequence-picker">
          <label>Household sequence
            <select value={activeSequence?.id || ""} onChange={e=>setSelectedSequence(e.target.value)}>
              {sequences.map(item=><option key={item.id} value={item.id}>{item.rows[0].HH_ID} · {item.rows[0].year}–{item.rows[item.rows.length-1].year} · {item.rows.length} observations</option>)}
            </select>
          </label>
          <span>{sequences.length} multi-observation sequences available in the exported inference file</span>
        </div>
        <div className="chart-grid two-col sequence-charts">
          <div className="sequence-chart">
            <h4>Observed feature values</h4>
            <div className="chart-wrap"><ResponsiveContainer width="100%" height={230}>
              <LineChart data={sequenceRows} margin={{top:10,right:12,left:-18,bottom:4}}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
                <XAxis dataKey="year" tickLine={false} axisLine={false}/>
                <YAxis domain={[0,1]} tickLine={false} axisLine={false}/>
                <Tooltip formatter={v=>Number(v).toFixed(3)}/>
                <Legend/>
                <Line type="monotone" dataKey="MDPS" stroke="#3976b9" strokeWidth={2.5} dot={{r:4}}/>
                <Line type="monotone" dataKey="VMDPS" stroke="#d18b38" strokeWidth={2.5} dot={{r:4}}/>
                <Line type="monotone" dataKey="dep_rate" name="Dependency rate" stroke="#2f9a83" strokeWidth={2.5} dot={{r:4}}/>
              </LineChart>
            </ResponsiveContainer></div>
          </div>
          <div className="sequence-chart">
            <h4>Posterior probability by year</h4>
            <div className="chart-wrap"><ResponsiveContainer width="100%" height={230}>
              <LineChart data={sequenceRows} margin={{top:10,right:12,left:-18,bottom:4}}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3"/>
                <XAxis dataKey="year" tickLine={false} axisLine={false}/>
                <YAxis domain={[0,1]} tickFormatter={v=>`${Math.round(v*100)}%`} tickLine={false} axisLine={false}/>
                <Tooltip formatter={v=>`${(Number(v)*100).toFixed(2)}%`}/>
                <Legend/>
                <Line type="monotone" dataKey="state_0_probability" name="State 0" stroke={COLORS[0]} strokeWidth={2.5} dot={{r:4}}/>
                <Line type="monotone" dataKey="state_1_probability" name="State 1" stroke={COLORS[1]} strokeWidth={2.5} dot={{r:4}}/>
                <Line type="monotone" dataKey="state_2_probability" name="State 2" stroke={COLORS[2]} strokeWidth={2.5} dot={{r:4}}/>
              </LineChart>
            </ResponsiveContainer></div>
          </div>
        </div>
        <div className="sequence-table-wrap"><table className="data-table">
          <thead><tr><th>Year</th><th>MDPS</th><th>VMDPS</th><th>Dependency</th><th>Viterbi state</th><th>P(State 0)</th><th>P(State 1)</th><th>P(State 2)</th></tr></thead>
          <tbody>{sequenceRows.map((row,i)=><tr key={`${row.sequence_id}-${row.year}-${i}`}>
            <td>{row.year}</td><td>{Number(row.MDPS).toFixed(3)}</td><td>{Number(row.VMDPS).toFixed(3)}</td><td>{Number(row.dep_rate).toFixed(3)}</td>
            <td><span className={`state-tag state-${row.hidden_state}`}>State {row.hidden_state}</span></td>
            <td>{(100*Number(row.state_0_probability)).toFixed(1)}%</td><td>{(100*Number(row.state_1_probability)).toFixed(1)}%</td><td>{(100*Number(row.state_2_probability)).toFixed(1)}%</td>
          </tr>)}</tbody>
        </table></div>
        <p className="small-note">Viterbi gives the most likely complete state path. The three posterior probabilities are calculated using Forward–Backward and sum to approximately 100% for each observation.</p>
      </> : <div className="empty-state">No multi-observation household sequences were found in the inference file.</div>}
    </Panel>

    <Panel title="9. How to evaluate this model responsibly" subtitle="What the current outputs can demonstrate—and what they cannot.">
      <div className="evaluation-grid">
        <div><b>What we can inspect</b><ul><li>EM convergence status and iteration count</li><li>State occupancy and emission profiles</li><li>Transition probabilities</li><li>Posterior uncertainty by observation</li><li>Household state paths across consecutive years</li></ul></div>
        <div><b>What we cannot claim</b><ul><li>Classification accuracy: no ground-truth hidden-state labels exist</li><li>Loan-default prediction: no default target or monthly repayment history is available</li><li>Causal impact of microcredit from this unsupervised HMM alone</li><li>Official poverty categories from the model-generated state IDs</li></ul></div>
      </div>
    </Panel>
  </>;
}

function Config({label,value}) {
  return <div className="config-item"><span>{label}</span><b>{value}</b></div>;
}

function Explorer({ records, query, setQuery, yearFilter, setYearFilter, stateFilter, setStateFilter, total }) {
  return <>
    <PageHeading eyebrow="RESULTS EXPLORER" title="Explore household-level inference"
      description="Filter the saved inference output by household ID, survey year and inferred hidden state. Showing up to 100 matching records."/>
    <Panel title="Filter observations" subtitle={`${total.toLocaleString("en-IN")} eligible observations available in the saved result file.`}>
      <div className="filter-row">
        <label className="search-field"><Search size={17}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search household ID"/></label>
        <select value={yearFilter} onChange={e=>setYearFilter(e.target.value)}><option value="all">All years</option>{[2016,2017,2018,2019].map(y=><option key={y}>{y}</option>)}</select>
        <select value={stateFilter} onChange={e=>setStateFilter(e.target.value)}><option value="all">All states</option><option value="0">State 0</option><option value="1">State 1</option><option value="2">State 2</option></select>
      </div>
      <div className="table-meta">Showing {records.length} matching rows (maximum 100).</div>
      <div className="data-table-wrap"><table className="data-table">
        <thead><tr><th>Household ID</th><th>Year</th><th>MDPS</th><th>VMDPS</th><th>Dependency</th><th>Hidden state</th><th>Profile</th></tr></thead>
        <tbody>{records.map((r,i)=><tr key={`${r.HH_ID}-${r.year}-${i}`}>
          <td className="mono">{r.HH_ID}</td><td>{r.year}</td><td>{Number(r.MDPS).toFixed(3)}</td><td>{Number(r.VMDPS).toFixed(3)}</td><td>{Number(r.dep_rate).toFixed(3)}</td>
          <td><span className={`state-tag state-${r.hidden_state}`}>State {r.hidden_state}</span></td><td>{r.vulnerability_profile}</td>
        </tr>)}</tbody>
      </table>{records.length===0 && <div className="empty-state">No records match these filters.</div>}</div>
    </Panel>
  </>;
}

export default App;