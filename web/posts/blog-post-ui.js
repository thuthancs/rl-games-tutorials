(function () {
  'use strict';

  function initScrollProgress() {
    var wrap = document.createElement('div');
    wrap.className = 'scroll-progress-wrap';
    wrap.setAttribute('aria-hidden', 'true');
    var bar = document.createElement('div');
    bar.className = 'scroll-progress-bar';
    wrap.appendChild(bar);
    document.body.insertBefore(wrap, document.body.firstChild);

    function updateProgress() {
      var docEl = document.documentElement;
      var scrollTop = window.scrollY || docEl.scrollTop;
      var scrollHeight = docEl.scrollHeight - docEl.clientHeight;
      var pct = scrollHeight > 0 ? Math.min(100, (scrollTop / scrollHeight) * 100) : 100;
      bar.style.width = pct + '%';
    }

    window.addEventListener('scroll', function () {
      requestAnimationFrame(updateProgress);
    }, { passive: true });
    updateProgress();
  }

  function initTocHighlight() {
    var content = document.querySelector('.article-content');
    var tocNav = document.querySelector('.article-toc .toc-nav');
    if (!content || !tocNav) return;

    var sections = content.querySelectorAll('section[id], [id]');
    var tocLinks = tocNav.querySelectorAll('a[href^="#"]');
    var activeId = null;
    var triggerTop = 120;

    function getActiveSection() {
      var current = null;
      for (var i = 0; i < sections.length; i++) {
        var el = sections[i];
        if (!el.id) continue;
        var top = el.getBoundingClientRect().top;
        if (top <= triggerTop) current = el.id;
      }
      if (current) return current;
      var first = null;
      for (var j = 0; j < sections.length; j++) {
        if (sections[j].id) { first = sections[j].id; break; }
      }
      return first;
    }

    function setActive(id) {
      if (id === activeId) return;
      activeId = id;
      tocLinks.forEach(function (a) {
        var href = a.getAttribute('href');
        var linkId = href && href.charAt(0) === '#' ? href.slice(1) : '';
        a.classList.toggle('toc-active', linkId === id);
      });
    }

    window.addEventListener('scroll', function () {
      requestAnimationFrame(function () { setActive(getActiveSection()); });
    }, { passive: true });
    setActive(getActiveSection());
  }

  function init() {
    initScrollProgress();
    initTocHighlight();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
