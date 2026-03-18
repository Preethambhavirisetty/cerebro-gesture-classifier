import { useState, useEffect, useRef } from "react";

const FONTS = [
  { id: "calibri", label: "Calibri", stack: "'Calibri', 'Gill Sans', 'Trebuchet MS', sans-serif" },
  { id: "times", label: "Times New Roman", stack: "'Times New Roman', Times, Georgia, serif" },
  { id: "georgia", label: "Georgia", stack: "Georgia, 'Times New Roman', serif" },
  { id: "garamond", label: "Garamond", stack: "'EB Garamond', Garamond, 'Times New Roman', serif" },
];

const OSTEP_CHAPTERS = {
  intro: {
    label: "Introduction",
    color: "#64748b",
    bg: "#f1f5f9",
    badge: "#475569",
    chapters: [
      { id: "ch1", num: 1, title: "Dialogue on Three Easy Pieces" },
      { id: "ch2", num: 2, title: "Introduction to Operating Systems" },
    ],
  },
  virtualization: {
    label: "Virtualization",
    color: "#1d4ed8",
    bg: "#eff6ff",
    badge: "#1e40af",
    chapters: [
      { id: "ch3", num: 3, title: "Dialogue on Virtualization" },
      { id: "ch4", num: 4, title: "Processes" },
      { id: "ch5", num: 5, title: "Process API" },
      { id: "ch6", num: 6, title: "Direct Execution" },
      { id: "ch7", num: 7, title: "CPU Scheduling" },
      { id: "ch8", num: 8, title: "Multi-level Feedback Queue (MLFQ)" },
      { id: "ch9", num: 9, title: "Lottery Scheduling" },
      { id: "ch10", num: 10, title: "Multi-CPU Scheduling" },
      { id: "ch11", num: 11, title: "Summary Dialogue on CPU" },
      { id: "ch12", num: 12, title: "Dialogue on Memory Virtualization" },
      { id: "ch13", num: 13, title: "Address Spaces" },
      { id: "ch14", num: 14, title: "Memory API" },
      { id: "ch15", num: 15, title: "Address Translation" },
      { id: "ch16", num: 16, title: "Segmentation" },
      { id: "ch17", num: 17, title: "Free Space Management" },
      { id: "ch18", num: 18, title: "Introduction to Paging" },
      { id: "ch19", num: 19, title: "Translation Lookaside Buffers (TLBs)" },
      { id: "ch20", num: 20, title: "Advanced Page Tables" },
      { id: "ch21", num: 21, title: "Swapping: Mechanisms" },
      { id: "ch22", num: 22, title: "Swapping: Policies" },
      { id: "ch23", num: 23, title: "Complete VM Systems" },
      { id: "ch24", num: 24, title: "Summary Dialogue on VM" },
    ],
  },
  concurrency: {
    label: "Concurrency",
    color: "#6d28d9",
    bg: "#f5f3ff",
    badge: "#5b21b6",
    chapters: [
      { id: "ch25", num: 25, title: "Dialogue on Concurrency" },
      { id: "ch26", num: 26, title: "Concurrency and Threads" },
      { id: "ch27", num: 27, title: "Thread API" },
      { id: "ch28", num: 28, title: "Locks" },
      { id: "ch29", num: 29, title: "Locked Data Structures" },
      { id: "ch30", num: 30, title: "Condition Variables" },
      { id: "ch31", num: 31, title: "Semaphores" },
      { id: "ch32", num: 32, title: "Concurrency Bugs" },
      { id: "ch33", num: 33, title: "Event-based Concurrency" },
      { id: "ch34", num: 34, title: "Summary Dialogue on Concurrency" },
    ],
  },
  persistence: {
    label: "Persistence",
    color: "#047857",
    bg: "#ecfdf5",
    badge: "#065f46",
    chapters: [
      { id: "ch35", num: 35, title: "Dialogue on Persistence" },
      { id: "ch36", num: 36, title: "I/O Devices" },
      { id: "ch37", num: 37, title: "Hard Disk Drives" },
      { id: "ch38", num: 38, title: "Redundant Disk Arrays (RAID)" },
      { id: "ch39", num: 39, title: "Files and Directories" },
      { id: "ch40", num: 40, title: "File System Implementation" },
      { id: "ch41", num: 41, title: "Fast File System (FFS)" },
      { id: "ch42", num: 42, title: "FSCK and Journaling" },
      { id: "ch43", num: 43, title: "Log-structured File System (LFS)" },
      { id: "ch44", num: 44, title: "Flash-based SSDs" },
      { id: "ch45", num: 45, title: "Data Integrity and Protection" },
      { id: "ch46", num: 46, title: "Summary Dialogue on Persistence" },
      { id: "ch47", num: 47, title: "Dialogue on Distribution" },
      { id: "ch48", num: 48, title: "Distributed Systems" },
      { id: "ch49", num: 49, title: "Network File System (NFS)" },
      { id: "ch50", num: 50, title: "Andrew File System (AFS)" },
      { id: "ch51", num: 51, title: "Summary Dialogue on Distribution" },
    ],
  },
  security: {
    label: "Security",
    color: "#b45309",
    bg: "#fffbeb",
    badge: "#92400e",
    chapters: [
      { id: "ch52", num: 52, title: "Dialogue on Security" },
      { id: "ch53", num: 53, title: "Introduction to Security" },
      { id: "ch54", num: 54, title: "Authentication" },
      { id: "ch55", num: 55, title: "Access Control" },
      { id: "ch56", num: 56, title: "Cryptography" },
      { id: "ch57", num: 57, title: "Distributed System Security" },
    ],
  },
};

const STORAGE_KEY = "ostep_notes_v2";
const FONT_KEY = "ostep_font_v1";

function loadNotes() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch { return {}; }
}
function saveNotes(n) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(n)); } catch {}
}

function NoteItem({ note, onDelete, onEdit, fontStack }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(note.text);
  const ref = useRef(null);
  useEffect(() => { if (editing && ref.current) ref.current.focus(); }, [editing]);

  function commit() {
    if (val.trim()) onEdit(note.id, val.trim());
    setEditing(false);
  }

  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: "8px",
      padding: "9px 11px", background: "#fff",
      borderRadius: "6px", marginBottom: "6px",
      border: "1px solid #e2e8f0",
      boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      transition: "box-shadow 0.15s",
      fontFamily: fontStack,
    }}
      onMouseEnter={e => e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.09)"}
      onMouseLeave={e => e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.04)"}
    >
      <span style={{ color: "#94a3b8", fontSize: "11px", marginTop: "3px", flexShrink: 0 }}>▸</span>
      {editing ? (
        <textarea
          ref={ref}
          value={val}
          onChange={e => setVal(e.target.value)}
          onBlur={commit}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commit(); }
            if (e.key === "Escape") { setVal(note.text); setEditing(false); }
          }}
          style={{
            flex: 1, background: "#f8fafc", border: "1px solid #cbd5e1",
            borderRadius: "4px", color: "#1e293b", padding: "5px 8px",
            fontSize: "14px", fontFamily: fontStack, resize: "vertical",
            minHeight: "42px", lineHeight: "1.6", outline: "none",
          }}
        />
      ) : (
        <span
          style={{ flex: 1, color: "#334155", fontSize: "14px", lineHeight: "1.7", cursor: "text", whiteSpace: "pre-wrap" }}
          onDoubleClick={() => setEditing(true)}
        >{note.text}</span>
      )}
      <div style={{ display: "flex", gap: "4px", flexShrink: 0, marginTop: "1px" }}>
        <button onClick={() => setEditing(true)} style={iconBtn("#e2e8f0", "#475569")} title="Edit">✎</button>
        <button onClick={() => onDelete(note.id)} style={iconBtn("#fee2e2", "#b91c1c")} title="Delete">×</button>
      </div>
    </div>
  );
}

function iconBtn(bg, color) {
  return {
    background: bg, border: "none", color, borderRadius: "4px",
    width: "23px", height: "23px", cursor: "pointer", fontSize: "13px",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0, padding: 0, lineHeight: 1, fontWeight: "600",
  };
}

function ChapterPanel({ chapter, section, notes, onAddNote, onDeleteNote, onEditNote, fontStack }) {
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);
  const chNotes = notes[chapter.id] || [];

  function add() {
    if (!input.trim()) return;
    onAddNote(chapter.id, input.trim());
    setInput("");
  }

  return (
    <div style={{
      borderRadius: "8px",
      border: "1px solid",
      borderColor: open ? section.color + "55" : "#e2e8f0",
      background: open ? section.bg : "#fff",
      overflow: "hidden",
      marginBottom: "7px",
      transition: "all 0.18s",
      boxShadow: open ? "0 2px 10px rgba(0,0,0,0.06)" : "0 1px 2px rgba(0,0,0,0.03)",
      fontFamily: fontStack,
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: "10px",
          padding: "11px 14px", background: "transparent", border: "none",
          cursor: "pointer", textAlign: "left",
        }}
      >
        <span style={{
          background: section.color, color: "#fff",
          borderRadius: "4px", fontSize: "10px", fontWeight: "700",
          padding: "2px 7px", flexShrink: 0, letterSpacing: "0.04em",
          fontFamily: "'Calibri', sans-serif",
        }}>Ch.{chapter.num}</span>
        <span style={{ color: "#1e293b", fontSize: "14px", fontWeight: open ? "600" : "400", flex: 1, fontFamily: fontStack }}>
          {chapter.title}
        </span>
        {chNotes.length > 0 && (
          <span style={{
            background: section.color + "20", borderRadius: "10px",
            fontSize: "11px", color: section.badge, padding: "1px 9px", fontWeight: "600",
            fontFamily: "'Calibri', sans-serif",
          }}>{chNotes.length}</span>
        )}
        <span style={{
          color: "#94a3b8", fontSize: "11px",
          transform: open ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s",
        }}>▶</span>
      </button>

      {open && (
        <div style={{ padding: "0 14px 14px" }}>
          {chNotes.length === 0 && (
            <p style={{ color: "#94a3b8", fontSize: "13px", fontStyle: "italic", margin: "0 0 10px", fontFamily: fontStack }}>
              No notes yet — add your first note below.
            </p>
          )}
          {chNotes.map(n => (
            <NoteItem key={n.id} note={n}
              onDelete={id => onDeleteNote(chapter.id, id)}
              onEdit={(id, text) => onEditNote(chapter.id, id, text)}
              fontStack={fontStack}
            />
          ))}
          <div style={{ display: "flex", gap: "8px", marginTop: "10px" }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); add(); } }}
              placeholder="Write a note… (Enter to save, Shift+Enter for new line)"
              style={{
                flex: 1, background: "#fff", border: "1px solid #cbd5e1",
                borderRadius: "6px", color: "#1e293b", padding: "8px 11px",
                fontSize: "14px", fontFamily: fontStack, resize: "vertical",
                minHeight: "40px", lineHeight: "1.6", outline: "none",
                boxShadow: "inset 0 1px 3px rgba(0,0,0,0.04)",
              }}
              onFocus={e => e.target.style.borderColor = section.color}
              onBlur={e => e.target.style.borderColor = "#cbd5e1"}
            />
            <button
              onClick={add}
              style={{
                background: section.color, border: "none", borderRadius: "6px",
                color: "#fff", fontWeight: "700", fontSize: "20px",
                width: "40px", flexShrink: 0, cursor: "pointer",
                alignSelf: "flex-end", height: "40px", lineHeight: 1,
              }}
            >+</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [notes, setNotes] = useState(loadNotes);
  const [activeSection, setActiveSection] = useState("intro");
  const [search, setSearch] = useState("");
  const [fontId, setFontId] = useState(() => {
    try { return localStorage.getItem(FONT_KEY) || "calibri"; } catch { return "calibri"; }
  });

  useEffect(() => { saveNotes(notes); }, [notes]);
  useEffect(() => { try { localStorage.setItem(FONT_KEY, fontId); } catch {} }, [fontId]);

  const font = FONTS.find(f => f.id === fontId) || FONTS[0];

  function addNote(chId, text) {
    setNotes(prev => ({ ...prev, [chId]: [...(prev[chId] || []), { id: Date.now().toString(), text }] }));
  }
  function deleteNote(chId, noteId) {
    setNotes(prev => ({ ...prev, [chId]: (prev[chId] || []).filter(n => n.id !== noteId) }));
  }
  function editNote(chId, noteId, text) {
    setNotes(prev => ({ ...prev, [chId]: (prev[chId] || []).map(n => n.id === noteId ? { ...n, text } : n) }));
  }

  const totalNotes = Object.values(notes).reduce((a, b) => a + b.length, 0);
  const searchMode = search.trim().length > 0;
  const searchLower = search.toLowerCase();

  let searchResults = [];
  if (searchMode) {
    for (const [secKey, sec] of Object.entries(OSTEP_CHAPTERS)) {
      for (const ch of sec.chapters) {
        const titleMatch = ch.title.toLowerCase().includes(searchLower);
        const noteMatch = (notes[ch.id] || []).some(n => n.text.toLowerCase().includes(searchLower));
        if (titleMatch || noteMatch) searchResults.push({ secKey, sec, ch });
      }
    }
  }

  const section = OSTEP_CHAPTERS[activeSection];

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", fontFamily: font.stack, display: "flex", flexDirection: "column" }}>

      {/* ── Header ── */}
      <div style={{
        background: "#fff",
        borderBottom: "1px solid #e2e8f0",
        padding: "0 20px",
        display: "flex", alignItems: "center", gap: "14px",
        position: "sticky", top: 0, zIndex: 100,
        boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
        height: "58px",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "9px" }}>
          <div style={{
            width: "32px", height: "32px", borderRadius: "7px",
            background: "linear-gradient(135deg, #1d4ed8 0%, #7c3aed 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "#fff", fontWeight: "800", fontSize: "13px",
            flexShrink: 0, fontFamily: "'Calibri', sans-serif",
          }}>OS</div>
          <div>
            <div style={{ fontSize: "15px", fontWeight: "700", color: "#0f172a", lineHeight: 1.15, fontFamily: font.stack }}>OSTEP Notes</div>
            <div style={{ fontSize: "9.5px", color: "#94a3b8", letterSpacing: "0.07em", fontFamily: "'Calibri', sans-serif" }}>THREE EASY PIECES</div>
          </div>
        </div>

        <div style={{ flex: 1 }} />

        {/* Font picker label */}
        <span style={{ fontSize: "11px", color: "#94a3b8", whiteSpace: "nowrap", fontFamily: "'Calibri', sans-serif" }}>Font:</span>

        {/* Font buttons */}
        {FONTS.map(f => (
          <button
            key={f.id}
            onClick={() => setFontId(f.id)}
            title={f.label}
            style={{
              padding: "4px 10px",
              borderRadius: "5px",
              border: fontId === f.id ? "2px solid #1d4ed8" : "1.5px solid #e2e8f0",
              background: fontId === f.id ? "#eff6ff" : "#fafafa",
              color: fontId === f.id ? "#1d4ed8" : "#64748b",
              fontSize: "12px",
              fontFamily: f.stack,
              cursor: "pointer",
              fontWeight: fontId === f.id ? "700" : "400",
              transition: "all 0.14s",
              whiteSpace: "nowrap",
              lineHeight: 1.5,
            }}
          >
            {f.label}
          </button>
        ))}

        {/* Search */}
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search chapters or notes…"
          style={{
            background: "#f1f5f9", border: "1px solid #e2e8f0",
            borderRadius: "7px", color: "#1e293b", padding: "6px 12px",
            fontSize: "13px", fontFamily: font.stack, outline: "none",
            width: "195px", transition: "border-color 0.15s",
          }}
          onFocus={e => e.target.style.borderColor = "#1d4ed8"}
          onBlur={e => e.target.style.borderColor = "#e2e8f0"}
        />

        {/* Total notes badge */}
        <div style={{
          background: "#f1f5f9", borderRadius: "6px",
          padding: "4px 11px", color: "#64748b", fontSize: "12px",
          whiteSpace: "nowrap", border: "1px solid #e2e8f0",
          fontFamily: "'Calibri', sans-serif",
        }}>
          {totalNotes} note{totalNotes !== 1 ? "s" : ""}
        </div>
      </div>

      <div style={{ display: "flex", flex: 1 }}>

        {/* ── Sidebar ── */}
        <div style={{
          width: "192px", flexShrink: 0,
          background: "#fff",
          borderRight: "1px solid #e2e8f0",
          padding: "14px 10px",
          position: "sticky", top: "58px",
          height: "calc(100vh - 58px)",
          overflowY: "auto",
        }}>
          <div style={{ fontSize: "10px", color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", padding: "0 6px 8px", fontFamily: "'Calibri', sans-serif" }}>
            Sections
          </div>
          {Object.entries(OSTEP_CHAPTERS).map(([key, sec]) => {
            const secNotes = sec.chapters.reduce((acc, ch) => acc + (notes[ch.id]?.length || 0), 0);
            const active = activeSection === key && !searchMode;
            return (
              <button
                key={key}
                onClick={() => { setActiveSection(key); setSearch(""); }}
                style={{
                  width: "100%", textAlign: "left",
                  padding: "9px 10px",
                  background: active ? sec.bg : "transparent",
                  border: active ? `1.5px solid ${sec.color}50` : "1.5px solid transparent",
                  borderRadius: "7px", cursor: "pointer", marginBottom: "3px",
                  display: "flex", flexDirection: "column", gap: "2px",
                  transition: "all 0.14s",
                }}
                onMouseEnter={e => { if (!active) e.currentTarget.style.background = "#f8fafc"; }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.background = active ? sec.bg : "transparent"; }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
                  <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: sec.color, flexShrink: 0 }} />
                  <span style={{
                    color: active ? sec.color : "#374151",
                    fontSize: "13px", fontWeight: active ? "700" : "500",
                    fontFamily: font.stack,
                  }}>{sec.label}</span>
                </div>
                <span style={{ color: "#9ca3af", fontSize: "10.5px", paddingLeft: "14px", fontFamily: "'Calibri', sans-serif" }}>
                  {sec.chapters.length} ch · {secNotes} notes
                </span>
              </button>
            );
          })}
        </div>

        {/* ── Main ── */}
        <div style={{ flex: 1, padding: "22px 28px", overflowY: "auto" }}>
          {searchMode ? (
            <>
              <div style={{ color: "#64748b", fontSize: "13px", marginBottom: "18px", fontFamily: font.stack }}>
                {searchResults.length === 0
                  ? `No results for "${search}"`
                  : `${searchResults.length} chapter${searchResults.length !== 1 ? "s" : ""} matching "${search}"`}
              </div>
              {searchResults.map(({ sec, ch }) => (
                <div key={ch.id} style={{ marginBottom: "10px" }}>
                  <div style={{ color: sec.color, fontSize: "10px", marginBottom: "4px", letterSpacing: "0.08em", textTransform: "uppercase", fontFamily: "'Calibri', sans-serif" }}>
                    {sec.label}
                  </div>
                  <ChapterPanel chapter={ch} section={sec} notes={notes}
                    onAddNote={addNote} onDeleteNote={deleteNote} onEditNote={editNote}
                    fontStack={font.stack}
                  />
                </div>
              ))}
            </>
          ) : (
            <>
              <div style={{ marginBottom: "22px", paddingBottom: "14px", borderBottom: `2px solid ${section.color}25` }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "5px" }}>
                  <div style={{ width: "4px", height: "24px", background: section.color, borderRadius: "3px" }} />
                  <h2 style={{
                    color: section.color, margin: 0, fontSize: "19px",
                    fontWeight: "700", fontFamily: font.stack,
                  }}>
                    {section.label}
                  </h2>
                  <span style={{ color: "#94a3b8", fontSize: "13px", fontFamily: "'Calibri', sans-serif" }}>
                    — {section.chapters.length} chapters
                  </span>
                </div>
                <p style={{ color: "#94a3b8", fontSize: "11.5px", margin: "0 0 0 14px", fontFamily: "'Calibri', sans-serif" }}>
                  Click a chapter to expand · Double-click any note to edit · Enter to save
                </p>
              </div>

              {section.chapters.map(ch => (
                <ChapterPanel
                  key={ch.id}
                  chapter={ch}
                  section={section}
                  notes={notes}
                  onAddNote={addNote}
                  onDeleteNote={deleteNote}
                  onEditNote={editNote}
                  fontStack={font.stack}
                />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}