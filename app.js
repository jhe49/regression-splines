// app.js

// Generate sample nonlinear data
const n = 200;
const X = Array.from({ length: n }, (_, i) => i * 10 / (n - 1));
const y_true = X.map(x => Math.sin(x) + 0.3 * x);
const noise = X.map(() => math.randomNormal(0, 0.5));
const y = y_true.map((yt, i) => yt + noise[i]);

// Utility: create Vandermonde matrix for polynomial regression
function polyFeatures(X, degree) {
  return X.map(x =>
    Array.from({ length: degree }, (_, j) => Math.pow(x, j + 1))
  );
}

// Utility: cubic spline basis expansion
function splineBasis(X, knots, degree = 3, natural = false) {
  const minX = Math.min(...X);
  const maxX = Math.max(...X);

  // Uniform knots
  const knotPositions = Array.from({ length: knots }, (_, i) =>
    minX + (i + 1) * (maxX - minX) / (knots + 1)
  );

  // Basis: poly part + truncated power functions
  let basis = X.map(x => [x, x ** 2, x ** 3]);
  knotPositions.forEach(knot => {
    basis.forEach((row, i) => {
      row.push(Math.max(0, X[i] - knot) ** 3);
    });
  });

  if (natural) {
    // impose linear constraints at boundaries by dropping two columns
    basis = basis.map(row => row.slice(0, -2));
  }

  return basis;
}

// Fit linear regression using normal equations
function fitLinear(Xmat, y) {
  const XmatM = math.matrix(Xmat);
  const yM = math.matrix(y);

  const Xt = math.transpose(XmatM);
  const beta = math.multiply(
    math.inv(math.multiply(Xt, XmatM)),
    math.multiply(Xt, yM)
  );

  return beta;
}

function predict(Xmat, beta) {
  return math.multiply(Xmat, beta).toArray();
}

// Update plots
function update() {
  const knots = parseInt(document.getElementById("knots").value);
  const degree = parseInt(document.getElementById("degree").value);
  const modelType = document.getElementById("model_type").value;

  let Xmat, beta, y_pred;

  if (modelType === "ols") {
    Xmat = X.map(x => [x]);
  } else if (modelType === "poly") {
    Xmat = polyFeatures(X, degree);
  } else if (modelType === "spline") {
    Xmat = splineBasis(X, knots, 3, false);
  } else if (modelType === "natural") {
    Xmat = splineBasis(X, knots, 3, true);
  }

  beta = fitLinear(Xmat, y);
  y_pred = predict(Xmat, beta);

  // Metrics
  const residuals = y.map((val, i) => val - y_pred[i]);
  const ssRes = residuals.reduce((acc, r) => acc + r * r, 0);
  const ssTot = y.reduce((acc, val) => acc + Math.pow(val - math.mean(y), 2), 0);
  const r2 = 1 - ssRes / ssTot;
  const rmse = Math.sqrt(ssRes / y.length);

  document.getElementById("metrics_table").innerHTML = `
    <tr><td>R²</td><td>${r2.toFixed(3)}</td></tr>
    <tr><td>RMSE</td><td>${rmse.toFixed(3)}</td></tr>
  `;

  // Fit plot
  const fitTrace = {
    x: X,
    y: y_pred,
    mode: "lines",
    name: "Fitted"
  };
  const dataTrace = { x: X, y: y, mode: "markers", name: "Data" };
  Plotly.newPlot("fit_plot", [dataTrace, fitTrace], {
    title: "Model Fit",
    xaxis: { title: "X" },
    yaxis: { title: "y" }
  });

  // Residual plot
  Plotly.newPlot("residual_plot", [
    { x: y_pred, y: residuals, mode: "markers", name: "Residuals" }
  ], {
    title: "Residual Plot",
    xaxis: { title: "Predicted" },
    yaxis: { title: "Residuals" }
  });
}

// Initial plot
update();
document.querySelectorAll("#knots, #degree, #model_type")
  .forEach(el => el.addEventListener("input", update));
