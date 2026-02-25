(function () {
  'use strict';

  var DIR_ARROWS = { upward: '\u2191', downward: '\u2193', leftward: '\u2190', rightward: '\u2192' };

  function neighbors(cell, gridSize) {
    var r = cell[0], c = cell[1];
    var out = [];
    [[-1,0],[1,0],[0,-1],[0,1]].forEach(function (d) {
      var nr = r + d[0], nc = c + d[1];
      if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize) out.push([nr, nc]);
    });
    return out;
  }

  function dirFrom(fromPos, toPos) {
    var dr = toPos[0] - fromPos[0], dc = toPos[1] - fromPos[1];
    if (dr === -1) return 'upward';
    if (dr === 1) return 'downward';
    if (dc === 1) return 'rightward';
    return 'leftward';
  }

  function generateConnectedPlacements(length, gridSize) {
    var cells = [];
    for (var r = 0; r < gridSize; r++) for (var c = 0; c < gridSize; c++) cells.push([r, c]);
    var placements = {};
    function pathKey(path) { return path.map(function (p) { return p[0] + ',' + p[1]; }).sort().join('|'); }
    function dfs(path) {
      if (path.length === length) {
        placements[pathKey(path)] = path.slice().map(function (p) { return p.slice(); });
        return;
      }
      var tail = path[path.length - 1];
      neighbors(tail, gridSize).forEach(function (nb) {
        if (path.every(function (p) { return p[0] !== nb[0] || p[1] !== nb[1]; })) dfs(path.concat([nb]));
      });
    }
    cells.forEach(function (cell) { dfs([cell.slice()]); });
    return Object.keys(placements).map(function (k) { return placements[k]; });
  }

  function headDirPairsForPlacement(shapeCells, gridSize) {
    var pairs = {};
    if (shapeCells.length === 1) {
      var only = shapeCells[0].slice();
      ['upward', 'downward', 'rightward', 'leftward'].forEach(function (d) {
        pairs[only[0] + ',' + only[1] + ',' + d] = true;
      });
      return Object.keys(pairs).map(function (k) {
        var parts = k.split(',');
        return [[parseInt(parts[0], 10), parseInt(parts[1], 10)], parts[2]];
      });
    }
    var shapeSet = {};
    shapeCells.forEach(function (p) { shapeSet[p[0] + ',' + p[1]] = true; });
    var n = shapeCells.length;
    function dfs(path) {
      if (path.length === n) {
        var head = path[0], second = path[1];
        /* Direction the head is facing = from body toward head (direction of movement) */
        var dir = dirFrom(second, head);
        pairs[head[0] + ',' + head[1] + ',' + dir] = true;
        return;
      }
      var tail = path[path.length - 1];
      neighbors(tail, gridSize).forEach(function (nb) {
        var key = nb[0] + ',' + nb[1];
        if (shapeSet[key] && path.every(function (p) { return p[0] !== nb[0] || p[1] !== nb[1]; })) dfs(path.concat([nb]));
      });
    }
    shapeCells.forEach(function (start) { dfs([start.slice()]); });
    return Object.keys(pairs).map(function (k) {
      var parts = k.split(',');
      return [[parseInt(parts[0], 10), parseInt(parts[1], 10)], parts[2]];
    });
  }

  function renderConfigGrid(config, gridSize, parentEl) {
    var bodySet = {};
    config.body.forEach(function (p) { bodySet[p[0] + ',' + p[1]] = true; });
    var head = config.headPos;
    var dir = config.headDir;
    var cellPx = 28;
    var totalPx = gridSize * cellPx;

    var grid = document.createElement('div');
    grid.className = 'valid-state-grid';
    grid.style.gridTemplateColumns = 'repeat(' + gridSize + ', 1fr)';
    grid.style.gridTemplateRows = 'repeat(' + gridSize + ', 1fr)';
    grid.style.width = totalPx + 'px';
    grid.style.height = totalPx + 'px';

    for (var r = 0; r < gridSize; r++) {
      for (var c = 0; c < gridSize; c++) {
        var cell = document.createElement('div');
        cell.className = 'valid-state-cell';
        if (head[0] === r && head[1] === c) {
          cell.classList.add('valid-state-head');
          cell.setAttribute('aria-label', 'head facing ' + dir);
          var arrow = document.createElement('span');
          arrow.className = 'valid-state-head-arrow';
          arrow.setAttribute('aria-hidden', 'true');
          arrow.textContent = DIR_ARROWS[dir] || dir;
          cell.appendChild(arrow);
        } else if (bodySet[r + ',' + c]) {
          cell.classList.add('valid-state-body');
        }
        grid.appendChild(cell);
      }
    }
    parentEl.appendChild(grid);
  }

  function runValidStatesDemo() {
    var gridSelect = document.getElementById('valid-states-grid-size');
    var lengthSelect = document.getElementById('valid-states-length');
    if (!gridSelect || !lengthSelect) return;
    var gridSize = parseInt(gridSelect.value, 10);
    var maxLen = gridSize * gridSize;
    var lengthOptions = lengthSelect.options;
    lengthOptions.length = 0;
    for (var len = 1; len <= maxLen; len++) {
      lengthOptions.add(new Option(len, len, false, len === 1));
    }
  }

  function onCompute() {
    var gridSize = parseInt(document.getElementById('valid-states-grid-size').value, 10);
    var length = parseInt(document.getElementById('valid-states-length').value, 10);
    var summaryEl = document.getElementById('valid-states-summary');
    var listEl = document.getElementById('valid-states-list');
    if (!summaryEl || !listEl) return;

    var placements = generateConnectedPlacements(length, gridSize);
    var configs = [];
    placements.forEach(function (placement) {
      headDirPairsForPlacement(placement, gridSize).forEach(function (pair) {
        configs.push({ headPos: pair[0], headDir: pair[1], body: placement });
      });
    });

    var numFoodSlots = gridSize * gridSize - length;
    var fullStateCount = configs.length * numFoodSlots;
    summaryEl.textContent = configs.length + ' (head, direction, body) configurations \u2192 ' + fullStateCount + ' full states (including all food positions).';

    var maxShow = 60;
    listEl.innerHTML = '';
    var toShow = configs.length <= maxShow ? configs : configs.slice(0, maxShow);
    toShow.forEach(function (c) {
      renderConfigGrid(c, gridSize, listEl);
    });
  }

  function initValidStatesDemo() {
    runValidStatesDemo();
    var computeBtn = document.getElementById('valid-states-compute');
    var gridSelect = document.getElementById('valid-states-grid-size');
    var lengthSelect = document.getElementById('valid-states-length');
    if (computeBtn) computeBtn.addEventListener('click', onCompute);
    if (gridSelect) gridSelect.addEventListener('change', function () { runValidStatesDemo(); onCompute(); });
    if (lengthSelect) lengthSelect.addEventListener('change', onCompute);
    onCompute();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initValidStatesDemo);
  } else {
    initValidStatesDemo();
  }
})();
