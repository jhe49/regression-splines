// --- Generate simulated nonlinear data ---
const N = 200;
const X = Array.from({length:N}, (_,i)=> i*10/(N-1));
const y_true = X.map(x => Math.sin(x) + 0.3*x);
const y = y_true.map(v => v + (Math.random()-0.5)*1); // add noise

// --- Polynomial regression helper ---
function polyFit(x, y, degree) {
    const Xmat = x.map(xi => Array.from({length: degree+1}, (_, j) => Math.pow(xi, j)));
    const Xt = math.transpose(Xmat);
    const XtX = math.multiply(Xt, Xmat);
    const Xty = math.multiply(Xt, y);
    const beta = math.multiply(math.inv(XtX), Xty);
    return beta;
}

function polyPredict(beta, x) {
    return x.map(xi => beta.reduce((sum, b, j) => sum + b * Math.pow(xi, j), 0));
}

// --- Compute metrics ---
function computeMetrics(y, y_pred) {
    const residuals = y.map((yi,i)=> yi - y_pred[i]);
    const ss_res = math.sum(residuals.map(r=>r**2));
    const mean_y = math.mean(y);
    const ss_tot = math.sum(y.map(yi=> (yi - mean_y)**2));
    const r2 = 1 - ss_res/ss_tot;
    const rmse = Math.sqrt(ss_res / y.length);
    return {residuals, r2, rmse};
}

// --- Cubic spline approximation ---
function cubicSpline(x, y, knots) {
    const x_min = Math.min(...x);
    const x_max = Math.max(...x);
    const step = (x_max - x_min)/(knots+1);
    const knot_positions = [];
    for(let i=1;i<=knots;i++) knot_positions.push(x_min + i*step);
    const all_knots = [x_min, ...knot_positions, x_max];

    const y_pred = x.map(xi => {
        let seg_idx = all_knots.length-2;
        for(let j=0;j<all_knots.length-1;j++){
            if(xi >= all_knots[j] && xi <= all_knots[j+1]) { seg_idx = j; break; }
        }
        const seg_start = all_knots[seg_idx];
        const seg_end = all_knots[seg_idx+1];
        const idxs = x.map((v, idx)=> (v>=seg_start && v<=seg_end) ? idx : -1).filter(idx=> idx>=0);
        const X_seg = idxs.map(idx=> x[idx]);
        const y_seg = idxs.map(idx=> y[idx]);
        const beta = polyFit(X_seg, y_seg, 3);
        return beta.reduce((sum,b,j)=> sum + b*Math.pow(xi,j),0);
    });
    return y_pred;
}

// --- Natural spline approximation ---
function naturalSpline(x, y, knots) {
    const y_pred = cubicSpline(x, y, knots);
    const x_min = Math.min(...x);
    const x_max = Math.max(...x);
    return y_pred.map((yi, i) => {
        if(x[i] < x_min) return y[0];
        if(x[i] > x_max) return y[y.length-1];
        return yi;
    });
}

// --- Update cubic vs natural spline plot ---
function updateSplineComparison() {
    const knots = parseInt(document.getElementById('knots').value);

    const y_cubic = cubicSpline(X, y, knots);
    const y_natural = naturalSpline(X, y, knots);

    const x_min = Math.min(...X);
    const x_max = Math.max(...X);
    const step = (x_max - x_min)/(knots+1);
    const knot_positions = Array.from({length:knots}, (_,i)=> x_min + (i+1)*step);

    const knot_lines = knot_positions.map(k => ({
        x: [k,k],
        y: [Math.min(...y), Math.max(...y)],
        mode: 'lines',
        line: {dash:'dot', color:'gray'},
        name:'Knot',
        showlegend:false
    }));

    Plotly.newPlot('spline_comparison_plot', [
        {x:X, y:y, mode:'markers', name:'Data'},
        {x:X, y:y_cubic, mode:'lines', name:'Cubic Spline', line:{color:'blue'}},
        {x:X, y:y_natural, mode:'lines', name:'Natural Spline', line:{color:'red', dash:'dash'}},
        ...knot_lines
    ], {
        title:'Cubic vs Natural Spline',
        margin:{t:40},
        xaxis:{title:'X'},
        yaxis:{title:'Y'}
    });
}

// --- Update main plots ---
function updatePlots() {
    const degree = parseInt(document.getElementById('degree').value);
    const modelType = document.getElementById('model_type').value;
    const knots = parseInt(document.getElementById('knots').value);

    let y_pred = [];

    if(modelType==='ols'){
        const beta = polyFit(X, y, 1);
        y_pred = polyPredict(beta, X);
    } else if(modelType==='poly'){
        const beta = polyFit(X, y, degree);
        y_pred = polyPredict(beta, X);
    } else if(modelType==='spline'){
        y_pred = cubicSpline(X, y, knots);
    } else if(modelType==='natural'){
        y_pred = naturalSpline(X, y, knots);
    }

    const {residuals, r2, rmse} = computeMetrics(y, y_pred);

    // Update metrics table
    const table = document.getElementById('metrics_table');
    table.innerHTML = `
        <tr><td>R²</td><td>${r2.toFixed(3)}</td></tr>
        <tr><td>RMSE</td><td>${rmse.toFixed(3)}</td></tr>
    `;

    // Knot positions for main fit plot
    const x_min = Math.min(...X);
    const x_max = Math.max(...X);
    const step = (x_max - x_min)/(knots+1);
    const knot_positions = Array.from({length:knots}, (_,i)=> x_min + (i+1)*step);

    const knot_lines = knot_positions.map(k => ({
        x: [k,k],
        y: [Math.min(...y), Math.max(...y)],
        mode:'lines',
        line:{dash:'dot', color:'gray'},
        name:'Knot',
        showlegend:false
    }));

    // --- Update fit plot ---
    Plotly.newPlot('fit_plot', [
        {x:X, y:y, mode:'markers', name:'Data'},
        {x:X, y:y_pred, mode:'lines', name:'Fitted'},
        ...knot_lines
    ], {margin: { t: 30 }});

    // --- Update residual plot with color by magnitude ---
    const maxResidual = Math.max(...residuals.map(r => Math.abs(r)));
    const colors = residuals.map(r => {
        const intensity = Math.min(1, Math.abs(r)/maxResidual);
        const red = Math.floor(255 * intensity);
        return `rgb(${red}, 0, 0)`;
    });

    Plotly.newPlot('residual_plot', [
        {
            x: y_pred,
            y: residuals,
            mode: 'markers',
            marker: {color: colors, size:8},
            name:'Residuals'
        }
    ], {
        margin:{t:30},
        xaxis:{title:'Predicted Y'},
        yaxis:{title:'Residuals'}
    });

    // --- Update cubic vs natural spline comparison ---
    updateSplineComparison();
}

// --- Event listeners ---
document.getElementById('degree').addEventListener('change', updatePlots);
document.getElementById('model_type').addEventListener('change', updatePlots);
document.getElementById('knots').addEventListener('input', updatePlots);

// --- Initial plot ---
updatePlots();


