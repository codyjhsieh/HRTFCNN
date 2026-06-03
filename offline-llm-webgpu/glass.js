// Liquid Glass interaction layer.
// Drives the material's dynamic behaviours faithfully to Apple's model:
//   1. Specular highlights track a virtual light source (pointer on desktop,
//      device tilt via gyroscope on mobile).
//   2. The wallpaper parallaxes against that same light, so the glass appears
//      to refract a world that moves behind it.
//   3. Press gives a liquid squash that springs back.
//   4. New chat bubbles are materialised as glass automatically.

const root = document.documentElement;
const registered = new Set();

// Virtual light position in viewport coordinates.
let lightX = window.innerWidth * 0.5;
let lightY = 0;
let targetX = lightX;
let targetY = lightY;
let tiltX = 0, tiltY = 0;          // current wallpaper parallax (px)
let tTiltX = 0, tTiltY = 0;        // target parallax

// ---- Register existing + future glass elements ----
function register(el) {
  if (!el || registered.has(el)) return;
  // ensure the three material sublayers exist
  ensureLayers(el);
  registered.add(el);
}
function ensureLayers(el) {
  const need = ["lg-tint", "lg-edge", "lg-spec"];
  // only add a layer if missing (panels in the HTML already have them)
  need.forEach((cls) => {
    if (![...el.children].some((c) => c.classList?.contains(cls))) {
      const span = document.createElement("span");
      span.className = cls;
      el.insertBefore(span, el.firstChild);
    }
  });
}

document.querySelectorAll(".lg").forEach(register);

// ---- Animation loop: ease the light + push CSS vars per element ----
function frame() {
  // critically-damped easing toward target
  lightX += (targetX - lightX) * 0.14;
  lightY += (targetY - lightY) * 0.14;
  tiltX += (tTiltX - tiltX) * 0.08;
  tiltY += (tTiltY - tiltY) * 0.08;

  root.style.setProperty("--tilt-x", tiltX.toFixed(2) + "px");
  root.style.setProperty("--tilt-y", tiltY.toFixed(2) + "px");

  for (const el of registered) {
    if (!el.isConnected) { registered.delete(el); continue; }
    const r = el.getBoundingClientRect();
    if (r.bottom < -80 || r.top > window.innerHeight + 80 || r.width === 0) continue;
    const mx = ((lightX - r.left) / r.width) * 100;
    const my = ((lightY - r.top) / r.height) * 100;
    el.style.setProperty("--mx", mx.toFixed(1) + "%");
    el.style.setProperty("--my", my.toFixed(1) + "%");
    // edge bevel angle follows the light for a believable rim glint
    const ang = Math.atan2(lightY - (r.top + r.height / 2), lightX - (r.left + r.width / 2));
    el.style.setProperty("--tilt-y", (ang * 57.3 + 90).toFixed(0) + "deg");
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---- Pointer drives the light (desktop / trackpad) ----
window.addEventListener("pointermove", (e) => {
  targetX = e.clientX;
  targetY = e.clientY;
  tTiltX = (e.clientX / window.innerWidth - 0.5) * -22;
  tTiltY = (e.clientY / window.innerHeight - 0.5) * -16;
}, { passive: true });

// ---- Gyroscope drives the light (mobile) ----
let gyroActive = false;
function onOrientation(ev) {
  if (ev.gamma == null && ev.beta == null) return;
  gyroActive = true;
  // gamma: left/right tilt [-90,90]; beta: front/back [-180,180]
  const g = Math.max(-45, Math.min(45, ev.gamma || 0));
  const b = Math.max(-30, Math.min(60, (ev.beta || 0) - 35));
  targetX = ((g + 45) / 90) * window.innerWidth;
  targetY = ((b + 30) / 90) * window.innerHeight;
  tTiltX = (g / 45) * -26;
  tTiltY = (b / 45) * -20;
}
function enableGyro() {
  const DOE = window.DeviceOrientationEvent;
  if (!DOE) return;
  if (typeof DOE.requestPermission === "function") {
    // iOS 13+ requires a user gesture to grant access
    DOE.requestPermission().then((state) => {
      if (state === "granted") window.addEventListener("deviceorientation", onOrientation, { passive: true });
    }).catch(() => {});
  } else {
    window.addEventListener("deviceorientation", onOrientation, { passive: true });
  }
}
// request gyro on first touch (the required user gesture)
window.addEventListener("touchstart", enableGyro, { once: true, passive: true });
// also fall back to touch position as a light source until/if gyro engages
window.addEventListener("touchmove", (e) => {
  if (gyroActive) return;
  const t = e.touches[0]; if (!t) return;
  targetX = t.clientX; targetY = t.clientY;
}, { passive: true });

// ---- Press: liquid squash (event-delegated) ----
function pressOn(e) {
  const el = e.target.closest(".lg-press");
  if (el && !el.disabled) el.classList.add("pressing");
}
function pressOff() {
  document.querySelectorAll(".lg-press.pressing").forEach((el) => el.classList.remove("pressing"));
}
document.addEventListener("pointerdown", pressOn, { passive: true });
document.addEventListener("pointerup", pressOff, { passive: true });
document.addEventListener("pointercancel", pressOff, { passive: true });
window.addEventListener("blur", pressOff);

// ---- Materialise chat bubbles as glass as they appear ----
const chat = document.getElementById("chat");
if (chat) {
  const obs = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1 || !node.classList?.contains("msg")) continue;
        if (node.classList.contains("system-note")) continue;
        node.classList.add("lg");
        if (node.classList.contains("user")) node.classList.add("lg-accent");
        register(node);
      }
    }
  });
  obs.observe(chat, { childList: true });
}

// ---- Entrance animation when chat/composer reveal ----
const revealTargets = ["composer", "chat"].map((id) => document.getElementById(id)).filter(Boolean);
const revealObs = new MutationObserver((muts) => {
  for (const m of muts) {
    const el = m.target;
    if (!el.classList.contains("hidden") && !el.dataset.entered) {
      el.dataset.entered = "1";
      el.animate(
        [
          { opacity: 0, transform: "translateY(16px) scale(0.97)", filter: "blur(8px)" },
          { opacity: 1, transform: "translateY(0) scale(1)", filter: "blur(0)" },
        ],
        { duration: 520, easing: "cubic-bezier(0.34,1.56,0.64,1)", fill: "backwards" }
      );
    }
  }
});
revealTargets.forEach((el) => revealObs.observe(el, { attributes: true, attributeFilter: ["class"] }));
