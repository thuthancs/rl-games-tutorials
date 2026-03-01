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

  function initCopyButtons() {
    var blocks = document.querySelectorAll('.article-content pre');
    blocks.forEach(function (pre) {
      var wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.setAttribute('aria-label', 'Copy code');
      btn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
          '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
          '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
        '</svg>';
      wrapper.appendChild(btn);

      btn.addEventListener('click', function () {
        var code = pre.querySelector('code');
        var text = code ? code.innerText : pre.innerText;
        navigator.clipboard.writeText(text).then(function () {
          btn.classList.add('copied');
          btn.setAttribute('aria-label', 'Copied!');
          btn.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
              '<polyline points="20 6 9 17 4 12"></polyline>' +
            '</svg>';
          setTimeout(function () {
            btn.classList.remove('copied');
            btn.setAttribute('aria-label', 'Copy code');
            btn.innerHTML =
              '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
                '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
                '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
              '</svg>';
          }, 2000);
        });
      });
    });
  }

  function init() {
    initScrollProgress();
    initTocHighlight();
    initCopyButtons();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
