/* Shared collapsible page-nav menu behaviour.
   Safe to load on any page — a no-op when there is no #navToggle / #headerNav.
   Markup contract:
     <button id="navToggle" class="nav-toggle" onclick="toggleNavMenu()"
             aria-label="Open page menu" aria-expanded="false"
             aria-controls="headerNav" aria-haspopup="true">☰</button>
     <nav id="headerNav" class="header-nav"> …links… </nav>
*/
(function () {
  const nav = () => document.getElementById('headerNav');
  const btn = () => document.getElementById('navToggle');

  function setNavMenu(open) {
    const n = nav(), b = btn();
    if (!n || !b) return;
    n.classList.toggle('open', open);
    b.setAttribute('aria-expanded', open ? 'true' : 'false');
    b.setAttribute('aria-label', open ? 'Close page menu' : 'Open page menu');
  }

  // Exposed for the inline onclick on the toggle button.
  window.toggleNavMenu = function () {
    const n = nav();
    if (n) setNavMenu(!n.classList.contains('open'));
  };

  // Close on outside click or selecting a link.
  document.addEventListener('click', e => {
    const n = nav();
    if (!n || !n.classList.contains('open')) return;
    if (e.target.closest('#headerNav a')) { setNavMenu(false); return; }
    if (!e.target.closest('#headerNav') && !e.target.closest('#navToggle')) setNavMenu(false);
  });

  // Close on Escape, returning focus to the toggle.
  document.addEventListener('keydown', e => {
    const n = nav();
    if (e.key === 'Escape' && n && n.classList.contains('open')) {
      setNavMenu(false);
      const b = btn();
      if (b) b.focus();
    }
  });
})();
