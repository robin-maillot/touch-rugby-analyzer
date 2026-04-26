var TR = {}; // var so it's a property of the global object (window in browsers, vm context in tests)

TR.APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzhMw4nQr2nVW0MLfPWVXRPXsvImerpzshQ5GnJ3873qQkGMz0bcJQdmTRXCxwygfFm/exec';

TR.secret = () => sessionStorage.getItem('password') || '';

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
