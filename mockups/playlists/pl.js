// Shared fixture + renderers for the playlist mockups. Events are drawn from
// three different matches on purpose: the whole point of a playlist is that it
// spans filters, so a one-game fixture would hide the problem being solved.
const COL = {'Try':'--try','Turnover':'--turn','Penalty Attack':'--patk',
             'Penalty Defence':'--pdef','Game Event':'--game','To Review':'--review'};
const col = t => `var(${COL[t] || '--dim'})`;

const EVENTS = [
  {t:'2:14', type:'Try',             name:'32 - Cut',    team:'France',  move:'32 - Cut',    g:'FRA v ENG'},
  {t:'4:47', type:'Turnover',        name:'6th Touch',   team:'England', move:'23 - Backdoor', g:'FRA v ENG'},
  {t:'7:02', type:'Try',             name:'23 - Backdoor', team:'France', move:'23 - Backdoor', g:'FRA v ENG'},
  {t:'9:38', type:'Penalty Attack',  name:'Offside',     team:'France',  move:'33 - Quicky', g:'FRA v ENG'},
  {t:'11:55',type:'Try',             name:'Scoop',       team:'England', move:'Scoop',       g:'FRA v ENG'},
  {t:'1:31', type:'Try',             name:'23 - Backdoor', team:'Wales', move:'23 - Backdoor', g:'WAL v SCO'},
  {t:'5:09', type:'Turnover',        name:'Ball Down',   team:'Scotland',move:'32 - Long',   g:'WAL v SCO'},
  {t:'8:44', type:'Try',             name:'French Flair',team:'Wales',   move:'French Flair',g:'WAL v SCO'},
  {t:'12:20',type:'Penalty Defence', name:'Early touch', team:'Scotland',move:'',            g:'WAL v SCO'},
  {t:'3:56', type:'Try',             name:'33 - Cut',    team:'Ireland', move:'33 - Cut',    g:'IRE v ITA'},
  {t:'6:41', type:'Turnover',        name:'Dummy Touch', team:'Italy',   move:'23 - Backdoor', g:'IRE v ITA'},
  {t:'10:08',type:'Try',             name:'23 - Backdoor', team:'Ireland',move:'23 - Backdoor',g:'IRE v ITA'},
];

// Three saved playlists, deliberately uneven: a big teaching set, a short
// opposition review, and an empty one you just made.
const PLAYLISTS = [
  {id:'p1', name:'Backdoor teaching set', note:'every backdoor, both ends', ids:[2,5,10,11,1]},
  {id:'p2', name:'France defence review', note:'for Tuesday',              ids:[3,0,6]},
  {id:'p3', name:'Nationals shortlist',   note:'empty',                    ids:[]},
];

// One event card. `extra` injects the option-specific affordance (a ＋, a
// checkbox, a drag handle) into the same grid, so the three mockups stay
// comparable row-for-row.
function evCard(e, i, opts) {
  opts = opts || {};
  return `<div class="ev ${opts.cls || ''}" style="border-left-color:${col(e.type)}"
    data-i="${i}" ${opts.onclick ? `onclick="${opts.onclick}"` : ''}>
    ${opts.lead || `<span class="ev-t">${e.t}</span>`}
    <span class="ev-main">
      <span class="ev-name">${e.name}</span>
      <span class="ev-meta">
        <span class="tag" style="background:${col(e.type)}22;color:${col(e.type)}">${e.type}</span>
        <span class="gmatch">${e.g}</span>
        <span class="team" style="font-size:.66rem;font-weight:600">${e.team}</span>
        ${e.move ? `<span class="move">${e.move}</span>` : ''}
      </span>
    </span>
    ${opts.trail || ''}
  </div>`;
}

function $(id) { return document.getElementById(id); }
function toast(msg) {
  let t = $('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    t.style.cssText = 'position:absolute;left:50%;bottom:18px;transform:translateX(-50%);' +
      'background:#1b2230;border:1px solid #3b82f6;color:#e9ecf3;padding:9px 15px;' +
      'border-radius:20px;font-size:.75rem;z-index:80;opacity:0;transition:opacity .18s;' +
      'pointer-events:none;box-shadow:0 6px 22px rgba(0,0,0,.6);white-space:nowrap';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = '1';
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.style.opacity = '0'; }, 1500);
}
