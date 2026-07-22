// Depends on js/config.js (TR namespace) and js/utils.js (TR.extractVideoId).
//
// Unified video player abstraction over two backends:
//   - 'youtube' : YouTube IFrame API (YT.Player)
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
    if (opts.provider === 'local')   return createLocal(opts);
    throw new Error('TR.player: unknown provider "' + opts.provider + '"');
  };

  // Pick the right provider for a _metadata entry.
  TR.player.providerFor = function (meta) {
    if (!meta) return null;
    if (meta.youtubelink) return 'youtube';
    return null;
  };

  // Whether the game has a GCS-backed source video the Cloud Run service can clip.
  TR.player.hasClip = function (meta) {
    return !!(meta && meta.gcsObject);
  };

  // Convert a game-clock time (seconds) to a position within the video, using
  // the game's stored videoOffset. Offset is ADDED to game time: a video that
  // starts at the 15th game-minute has videoOffset = -900, so game 20:00 → 300s
  // into the video. A negative result means the moment happened before the
  // recording started (no footage for it).
  TR.player.videoTimeAt = function (meta, seconds) {
    return (Number(seconds) || 0) + ((meta && Number(meta.videoOffset)) || 0);
  };

  // Whether the game's video actually covers this game-clock moment.
  TR.player.hasVideoAt = function (meta, seconds) {
    return TR.player.videoTimeAt(meta, seconds) >= 0;
  };

  // Build a deep-link URL that opens the video at a given moment.
  // The 5-second lookback matches the existing per-event seek behavior.
  // Returns '' when the game has no video OR the moment precedes the recording.
  TR.player.seekLink = function (meta, seconds) {
    const videoTime = TR.player.videoTimeAt(meta, seconds);
    if (videoTime < 0) return '';                       // before the video starts
    const t = Math.max(0, Math.floor(videoTime - 5));
    const vid = TR.extractVideoId(meta && meta.youtubelink);
    return vid ? `https://www.youtube.com/watch?v=${vid}&t=${t}s` : '';
  };
})();
