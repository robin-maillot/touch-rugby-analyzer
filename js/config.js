var TR = {}; // var so it's a property of the global object (window in browsers, vm context in tests)

TR.APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhMw4nQr2nVW0MLfPWVXRPXsvImerpzshQ5GnJ3873qQkGMz0bcJQdmTRXCxwygfFm/exec';
TR.CLIP_SERVICE_URL = 'https://m30-clipper-1070277967282.europe-west1.run.app';

TR.secret = () => sessionStorage.getItem('password') || '';

TR.role = () => {
  const p = sessionStorage.getItem('password');
  return p === 'm30-admin' ? 'admin'
       : p === 'm30-staff' ? 'staff'
       : p === 'm30'       ? 'viewer' : 'anon';
};

// Umami pages set data-auto-track="false" so we send pageviews manually with
// role merged in. identify() tags the session; track() emits the pageview.
TR.umamiIdentify = (extra) => {
  const data = Object.assign({ role: TR.role() }, extra || {});
  const send = () => {
    if (!window.umami) return;
    if (window.umami.identify) window.umami.identify(data);
    if (window.umami.track) window.umami.track(props => Object.assign({}, props, data));
  };
  if (window.umami && window.umami.track) send();
  else if (typeof window.addEventListener === 'function') window.addEventListener('load', send);
};

// Redirect to index.html if the stored password doesn't meet the required role.
// role: 'viewer' | 'staff' | 'admin'
TR.auth = (role) => {
  const p = sessionStorage.getItem('password');
  const ok = {
    viewer: p === 'm30' || p === 'm30-staff' || p === 'm30-admin',
    staff:  p === 'm30-staff' || p === 'm30-admin',
    admin:  p === 'm30-admin',
  };
  if (!ok[role]) window.location.replace('index.html');
};

// Emit a single role-tagged pageview as soon as config.js loads.
TR.umamiIdentify();
