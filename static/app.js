const btnCamera = document.getElementById("btnCamera");
const btnShot   = document.getElementById("btnShot");
const btnRun    = document.getElementById("btnRun");
const fileInput = document.getElementById("fileInput");

const video      = document.getElementById("video");
const canvas     = document.getElementById("canvas");
const previewImg = document.getElementById("previewImg");
const resultImg  = document.getElementById("resultImg");
const list       = document.getElementById("list");
const hint       = document.getElementById("hint");

const confInp   = document.getElementById("conf");
const distInp   = document.getElementById("dist");
const pconfInp  = document.getElementById("pconf");
const aratioInp = document.getElementById("aratio");

const btnRefresh  = document.getElementById("btnRefresh");
const stTotal     = document.getElementById("stTotal");
const stWith      = document.getElementById("stWith");
const stBags      = document.getElementById("stBags");
const stRate      = document.getElementById("stRate");
const historyList = document.getElementById("historyList");

let stream = null;
let selectedBlob = null;

let lastSource = "upload";
let lastFilename = "image.jpg";

function setHint(text) { hint.textContent = text; }
function clearResult() { resultImg.src = ""; list.innerHTML = ""; }

function dictToStr(d) {
  if (!d || typeof d !== "object") return "-";
  const entries = Object.entries(d);
  if (entries.length === 0) return "-";
  return entries.map(([k, v]) => `${k}: ${v}`).join(", ");
}

function showPreviewImageFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  previewImg.src = url;
  previewImg.style.display = "block";
  video.style.display = "none";
  canvas.style.display = "none";
  btnRun.disabled = false;
  setHint("Готово: можно нажать «Запустить обработку»");
}

function showVideo() {
  video.style.display = "block";
  canvas.style.display = "none";
  previewImg.style.display = "none";
  setHint("Нажмите «Снимок», чтобы зафиксировать кадр");
}

async function startCamera() {
  clearResult();
  selectedBlob = null;
  btnRun.disabled = true;

  lastSource = "camera";
  lastFilename = "camera.jpg";

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  showVideo();
  btnShot.disabled = false;
}

function takeSnapshot() {
  if (!stream) return;

  const ctx = canvas.getContext("2d");
  const w = 640, h = 480;
  canvas.width = w; canvas.height = h;
  ctx.drawImage(video, 0, 0, w, h);

  canvas.toBlob((blob) => {
    selectedBlob = blob;

    stream.getTracks().forEach(t => t.stop());
    stream = null;
    btnShot.disabled = true;

    showPreviewImageFromBlob(blob);
  }, "image/jpeg", 0.95);
}

async function fetchHistory() {
  const resp = await fetch("/history?limit=30");
  if (!resp.ok) return;
  const data = await resp.json();

  const s = data.summary || {};
  stTotal.textContent = s.total_requests ?? 0;
  stWith.textContent  = s.requests_with_unattended ?? 0;
  stBags.textContent  = s.total_unattended_found ?? 0;
  stRate.textContent  = ((s.unattended_rate ?? 0) * 100).toFixed(1) + "%";

  const items = data.items || [];
  if (items.length === 0) {
    historyList.innerHTML = `<div class="empty">История пуста</div>`;
    return;
  }

  historyList.innerHTML = items.map(x => {
    const imgUrl = `/history/image/${x.id}`;

    // новые поля с бэкенда (если нет — fallback)
    const bagCounts = x.bag_counts_by_type || {};
    const unCounts  = x.unattended_counts_by_type || {};

    return `
      <div class="item">
        <div class="item-title">
          ${x.ts} <span class="muted">(${x.source})</span>
          <span class="muted"> • unattended=${x.unattended_count}</span>
        </div>

        <div class="muted">
          Найдено: people=${x.persons_count ?? 0} • baggage=${x.bags_count ?? 0} • unattended=${x.unattended_count ?? 0}
        </div>

        <div class="muted">baggage types: ${dictToStr(bagCounts)}</div>
        <div class="muted">unattended types: ${dictToStr(unCounts)}</div>

        <img src="${imgUrl}"
             style="width:100%; margin-top:8px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);" />
      </div>
    `;
  }).join("");
}

btnRefresh?.addEventListener("click", fetchHistory);
window.addEventListener("load", fetchHistory);

fileInput.addEventListener("change", () => {
  clearResult();
  const f = fileInput.files?.[0];
  if (!f) return;

  lastSource = "upload";
  lastFilename = f.name || "upload.jpg";

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    btnShot.disabled = true;
  }

  selectedBlob = f;
  showPreviewImageFromBlob(f);
});

btnCamera.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (e) {
    setHint("Не удалось открыть камеру. Проверьте разрешения браузера.");
  }
});

btnShot.addEventListener("click", () => takeSnapshot());

btnRun.addEventListener("click", async () => {
  if (!selectedBlob) return;

  clearResult();
  setHint("Обработка...");

  const fd = new FormData();
  fd.append("file", selectedBlob, lastFilename);

  fd.append("source", lastSource);
  fd.append("conf", confInp.value);
  fd.append("dist_thr", distInp.value);
  fd.append("person_conf", pconfInp.value);
  fd.append("min_area_ratio", aratioInp.value);

  const resp = await fetch("/predict", { method: "POST", body: fd });
  if (!resp.ok) {
    setHint("Ошибка бэкенда /predict");
    return;
  }
  const data = await resp.json();

  resultImg.src = "data:image/png;base64," + data.annotated_png_base64;

  const counts = data.counts || { person: 0, baggage: 0, unattended: 0 };
  const bagCounts = data.bag_counts_by_type || {};
  const unCounts  = data.unattended_counts_by_type || {};

  // Сводка по кадру: люди/багаж/забытый багаж
  let html = `
    <div class="item">
      <div class="item-title">Найдено на изображении</div>
      <div class="muted">Люди: ${counts.person} • Багаж: ${counts.baggage} • Забытый багаж: ${counts.unattended}</div>
      <div class="muted">baggage types: ${dictToStr(bagCounts)}</div>
      <div class="muted">unattended types: ${dictToStr(unCounts)}</div>
    </div>
  `;

  const items = data.unattended || [];
  if (items.length === 0) {
    html += `<div class="empty">Забытый багаж не обнаружен (по текущим порогам).</div>`;
  } else {
    html += items.map(x => `
      <div class="item">
        <div class="item-title">${x.bag_type} <span class="muted">conf=${x.conf.toFixed(2)}</span></div>
        <div class="muted">bbox: [${x.x1.toFixed(0)}, ${x.y1.toFixed(0)}, ${x.x2.toFixed(0)}, ${x.y2.toFixed(0)}]
        ${x.min_norm_dist !== null ? ` • dist=${x.min_norm_dist.toFixed(2)}` : ""}</div>
      </div>
    `).join("");
  }

  list.innerHTML = html;

  setHint("Готово");
  await fetchHistory();
});
