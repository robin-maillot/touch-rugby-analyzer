// Depends on js/config.js (TR namespace) and js/utils.js (TR.extractVideoId).
//
// Unified video player abstraction over three backends:
//   - 'youtube' : YouTube IFrame API (YT.Player)
//   - 'stream'  : Cloudflare Stream player (Stream(iframe))
//   - 'local'   : an existing HTMLVideoElement (e.g. a user-selected local file)
//
// All providers return the same shape from TR.player.create(opts):
//   { provider, ready, getTime, getDuration, isPaused, play, pause,
//     seek, setRate, destroy }
//
// Event callbacks:
//   onReady()           — fired once when the underlying player can accept commands.
//   onStateChange(s)    — 'playing' | 'paused' | 'ended'

TR.player = TR.player || {};

(function () {
  // ── YouTube ───────────────────────────────────────────────────
  function createYouTube(opts) {
    const { container, videoId, onReady, onStateChange } = opts;
    container.innerHTML = '<div></div>';
    const inner = container.firstElementChild;
    let yt = null;
    let ready = false;
    const init = () => {
      yt = new YT.Player(inner, {
        videoId,
        width: '100%',
        height: '100%',
        playerVars: { rel: 0, modestbranding: 1 },
        events: {
          onReady: () => { ready = true; if (onReady) onReady(); },
          onStateChange: ev => {
            if (!onStateChange) return;
            if      (ev.data === YT.PlayerState.PLAYING) onStateChange('playing');
            else if (ev.data === YT.PlayerState.PAUSED)  onStateChange('paused');
            else if (ev.data === YT.PlayerState.ENDED)   onStateChange('ended');
          }
        }
      });
    };
    if (window.YT && YT.Player) init();
    else {
      const queue = (window._trYTQueue = window._trYTQueue || []);
      queue.push(init);
      // Multiple TR.player instances must coexist; chain through any
      // existing callback rather than overwriting.
      const prior = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => {
        if (typeof prior === 'function' && prior !== window.onYouTubeIframeAPIReady) prior();
        (window._trYTQueue || []).forEach(fn => fn());
        window._trYTQueue = [];
      };
      if (!document.getElementById('ytApiScript')) {
        const tag = document.createElement('script');
        tag.id  = 'ytApiScript';
        tag.src = 'https://www.youtube.com/iframe_api';
        document.head.appendChild(tag);
      }
    }
    return {
      provider: 'youtube',
      get ready() { return ready; },
      getTime:     () => yt ? yt.getCurrentTime() : 0,
      getDuration: () => (yt && typeof yt.getDuration === 'function') ? yt.getDuration() : 0,
      isPaused:    () => !yt || yt.getPlayerState() !== YT.PlayerState.PLAYING,
      play:        () => { if (yt) yt.playVideo(); },
      pause:       () => { if (yt) yt.pauseVideo(); },
      seek:        (t) => { if (yt) yt.seekTo(t, true); },
      setRate:     (r) => { if (yt) yt.setPlaybackRate(r); },
      destroy:     () => { if (yt) { yt.destroy(); yt = null; } container.innerHTML = ''; ready = false; }
    };
  }

  // ── Cloudflare Stream ─────────────────────────────────────────
  // The Stream SDK exposes a Player object on an <iframe>; properties
  // (currentTime, duration, playbackRate, paused) and methods (play/pause)
  // mirror HTMLMediaElement. Events: 'loadedmetadata', 'play', 'pause',
  // 'ended', 'seeked', etc.
  function createStream(opts) {
    const { container, uid, signedToken, onReady, onStateChange } = opts;
    container.innerHTML = '';
    const iframe = document.createElement('iframe');
    iframe.src = `https://iframe.videodelivery.net/${signedToken || uid}`;
    iframe.style.width  = '100%';
    iframe.style.height = '100%';
    iframe.style.border = '0';
    iframe.allow = 'accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;';
    iframe.allowFullscreen = true;
    container.appendChild(iframe);

    let cf = null;
    let ready = false;
    let playing = false;
    function attach() {
      cf = window.Stream(iframe);
      cf.addEventListener('loadedmetadata', () => {
        ready = true;
        if (onReady) onReady();
      });
      cf.addEventListener('play',  () => { playing = true;  if (onStateChange) onStateChange('playing'); });
      cf.addEventListener('pause', () => { playing = false; if (onStateChange) onStateChange('paused');  });
      cf.addEventListener('ended', () => { playing = false; if (onStateChange) onStateChange('ended');   });
    }
    if (window.Stream) attach();
    else {
      const queue = (window._trStreamQueue = window._trStreamQueue || []);
      queue.push(attach);
      const existing = document.getElementById('cfStreamSdk');
      if (existing) {
        existing.addEventListener('load', () => {
          (window._trStreamQueue || []).forEach(fn => fn());
          window._trStreamQueue = [];
        });
      } else {
        const s = document.createElement('script');
        s.id  = 'cfStreamSdk';
        s.src = 'https://embed.cloudflarestream.com/embed/sdk.latest.js';
        s.onload = () => {
          (window._trStreamQueue || []).forEach(fn => fn());
          window._trStreamQueue = [];
        };
        document.head.appendChild(s);
      }
    }

    return {
      provider: 'stream',
      get ready() { return ready; },
      getTime:     () => cf ? (cf.currentTime || 0) : 0,
      getDuration: () => cf ? (cf.duration    || 0) : 0,
      isPaused:    () => !playing,
      play:        () => { if (cf) cf.play(); },
      pause:       () => { if (cf) cf.pause(); },
      seek:        (t) => { if (cf) cf.currentTime = t; },
      setRate:     (r) => { if (cf) cf.playbackRate = r; },
      destroy:     () => { container.innerHTML = ''; cf = null; ready = false; }
    };
  }

  // ── Local <video> element ─────────────────────────────────────
  function createLocal(opts) {
    const { videoElement, onReady, onStateChange } = opts;
    const vid = videoElement;
    let ready = false;
    const onMeta = () => { ready = true; if (onReady) onReady(); };
    const onPlay  = () => onStateChange && onStateChange('playing');
    const onPause = () => onStateChange && onStateChange('paused');
    const onEnded = () => onStateChange && onStateChange('ended');
    vid.addEventListener('loadedmetadata', onMeta);
    vid.addEventListener('play',  onPlay);
    vid.addEventListener('pause', onPause);
    vid.addEventListener('ended', onEnded);
    return {
      provider: 'local',
      get ready() { return ready; },
      getTime:     () => vid.currentTime,
      getDuration: () => vid.duration,
      isPaused:    () => vid.paused,
      play:        () => vid.play(),
      pause:       () => vid.pause(),
      seek:        (t) => { vid.currentTime = t; },
      setRate:     (r) => { vid.playbackRate = r; },
      destroy:     () => {
        vid.removeEventListener('loadedmetadata', onMeta);
        vid.removeEventListener('play',  onPlay);
        vid.removeEventListener('pause', onPause);
        vid.removeEventListener('ended', onEnded);
        try { vid.pause(); } catch (e) {}
        vid.removeAttribute('src');
        vid.load();
        ready = false;
      }
    };
  }

  TR.player.create = function (opts) {
    if (opts.provider === 'youtube') return createYouTube(opts);
    if (opts.provider === 'stream')  return createStream(opts);
    if (opts.provider === 'local')   return createLocal(opts);
    throw new Error('TR.player: unknown provider "' + opts.provider + '"');
  };

  // Pick the right provider for a _metadata entry.
  // Explicit `videoprovider` wins so admins can override; otherwise infer
  // from which fields are populated (streamuid > youtubelink).
  TR.player.providerFor = function (meta) {
    if (!meta) return null;
    const p = (meta.videoprovider || '').toLowerCase();
    if (p === 'youtube' || p === 'stream') return p;
    if (meta.streamuid)   return 'stream';
    if (meta.youtubelink) return 'youtube';
    return null;
  };

  // Build a deep-link URL that opens the video at a given moment.
  // Used by games.html / viewer.html to produce shareable seek links.
  // The 5-second lookback matches the existing per-event seek behavior.
  TR.player.seekLink = function (meta, seconds) {
    const provider = TR.player.providerFor(meta);
    const t = Math.max(0, Math.floor((Number(seconds) || 0) - 5));
    if (provider === 'youtube') {
      const vid = TR.extractVideoId(meta && meta.youtubelink);
      return vid ? `https://www.youtube.com/watch?v=${vid}&t=${t}s` : '';
    }
    if (provider === 'stream' && meta && meta.streamuid) {
      // The watch.cloudflarestream.com host redirects to the customer
      // subdomain and drops the query string in the process, so startTime is
      // lost. The iframe.videodelivery.net player respects startTime directly
      // and works for public videos without needing a customer-code lookup.
      return `https://iframe.videodelivery.net/${meta.streamuid}?startTime=${t}`;
    }
    return '';
  };
})();
