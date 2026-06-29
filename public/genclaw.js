(function () {
  function fallbackWriteText(text) {
    return new Promise(function (resolve, reject) {
      try {
        var textarea = document.createElement("textarea");
        textarea.value = text == null ? "" : String(text);
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.top = "-1000px";
        textarea.style.left = "-1000px";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        var ok = document.execCommand("copy");
        document.body.removeChild(textarea);
        if (ok) {
          resolve();
        } else {
          reject(new Error("execCommand copy failed"));
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  try {
    var existing = navigator.clipboard;
    var originalWriteText = existing && existing.writeText
      ? existing.writeText.bind(existing)
      : null;
    var clipboard = {
      writeText: function (text) {
        if (!originalWriteText) {
          return fallbackWriteText(text);
        }
        return originalWriteText(text).catch(function () {
          return fallbackWriteText(text);
        });
      }
    };
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: clipboard
    });
  } catch (error) {
    window.__genclawClipboardFallbackError = String(error);
  }

  function looksLikeCopyButton(button) {
    if (!button || !button.matches || !button.matches(
      'button,[role="button"],[data-testid*="copy" i],[aria-label*="copy" i],[title*="copy" i]'
    )) {
      return false;
    }
    var text = [
      button.getAttribute("aria-label") || "",
      button.getAttribute("title") || "",
      button.getAttribute("data-testid") || "",
      button.className || "",
      button.textContent || "",
      button.innerHTML || ""
    ].join(" ").toLowerCase();
    return text.indexOf("copy") >= 0
      || text.indexOf("clipboard") >= 0
      || text.indexOf("lucide-copy") >= 0;
  }

  function hideCopyButtons(root) {
    var scope = root && root.querySelectorAll ? root : document;
    var buttons = scope.querySelectorAll(
      'button,[role="button"],[data-testid*="copy" i],[aria-label*="copy" i],[title*="copy" i]'
    );
    buttons.forEach(function (button) {
      if (looksLikeCopyButton(button)) {
        button.classList.add("genclaw-hide-copy");
        button.style.display = "none";
      }
    });
  }

  hideCopyButtons(document);
  var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (node) {
        if (node.nodeType === 1) {
          hideCopyButtons(node);
          if (looksLikeCopyButton(node)) {
            node.classList.add("genclaw-hide-copy");
            node.style.display = "none";
          }
        }
      });
    });
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
