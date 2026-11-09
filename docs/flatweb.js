//
// #INFO: A Simple JS for WebBook Interface
//
// -----------------------------------------------------------------------------
//
// Author: DTMc
//
// This file is part of lms.
//
// lms is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// lms is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// -----------------------------------------------------------------------------
//
// Define any relevant constants and variables here.
//
// Variable to store all the keyboard navigation available

var keyboardHelpMenu = {
  h: "Enable/Disable Help Menu",
  l: "Visit Lecture Structure",
  q: "Kill Pop-up Window",
  n: "Next Chapter",
  p: "Previous Chapter",
  b: "Bottom of Page",
  u: "Top of Page",
  i: "Go to Entry Point",
  t: "Enable/Disable Table of Contents",
  "Ctrl + k": "Search within the website",
};

// Defines the sizes of the three display categories.
var const_viewScreenGuide = {
  "Max-Computer": "1420",
  Computer: "1200",
  Tablet: "1000",
  Phone: "600",
};

var moodMenu = {
  Visit: [
    {
      "Central GitHub": [
        {
          key: "nil",
          command:
            "window.location.href = $('.toc-links').find('a').eq(0).attr('href')",
        },
      ],
      "Google Scholar": [
        {
          key: "nil",
          command:
            "window.location.href = $('.toc-links').find('a').eq(1).attr('href')",
        },
      ],
      ResearchGate: [
        {
          key: "nil",
          command:
            "window.location.href = $('.toc-links').find('a').eq(2).attr('href')",
        },
      ],

      "Sep-i": [],

      "Proposed Thesis Topics": [
        {
          key: "nil",
          command:
            "window.location.href = $('.toc-links').find('a').eq(3).attr('href')",
        },
      ],
      "mciDoc LaTeX Documentation": [
        {
          key: "nil",
          command:
            "window.location.href = $('.toc-links').find('a').eq(4).attr('href')",
        },
      ],

      "Sep-ii": [],

      "Visit Code NoteBook": [
        {
          key: "c",
          command: "menu__VisitCorg()",
        },
      ],
    },
  ],

  Overview: [
    {
      "Help Menu": [
        {
          key: "h",
          command: "toggleHelp()",
        },
      ],

      "Change Theme": [
        {
          key: "nil",
          command: "menu__switchTheme()",
        },
      ],

      "Reset Page": [
        {
          key: "r",
          command: "menu__refreshPage()",
        },
      ],

      "View Lecture Information": [
        {
          key: "l",
          command: "menu__showLectureStructure()",
        },
      ],

      "Sep-iii": [],

      "Toggle CurrentPage TOC": [
        {
          key: ".",
          command: "menu__toggleCurrentPageTOC()",
        },
      ],

      "Sep-ii": [],

      "Reader View": [
        {
          key: "w",
          command: "view__reader()",
        },
      ],

      "About WebBook": [
        {
          key: "h",
          command: "toggleHelp()",
        },
      ],

      "Quit WebBook": [
        {
          key: "Ctrl + q",
          command: "menu__killCurrentTab()",
        },
      ],
    },
  ],
  Navigate: [
    {
      "Top of Current Page": [
        {
          key: "u",
          command: "navigate__gotoTopOfPage()",
        },
      ],

      "Bottom of Current Page": [
        {
          key: "b",
          command: "navigate__toBottomOfPage()",
        },
      ],

      "Sep-i": [],

      "To Next Topic": [
        {
          key: "n",
          command: "navigate__gotoNextChapter()",
        },
      ],

      "To Previous Topic": [
        {
          key: "p",
          command: "navigate__gotoPrevChapter()",
        },
      ],

      "Sep-ii": [],

      "To Chapter Page": [
        {
          key: "d",
          command: "navigate__gotoChapterPage()",
        },
      ],

      "To Index Page": [
        {
          key: "i",
          command: "navigate__gotoIndexPage()",
        },
      ],
    },
  ],
};

//
// A List of all relevant documentation sites for possible querying
var DocumentationSites = {
  "pytorch-2.13": "https://docs.pytorch.org/docs/2.13/generated/",
  "pytorch-2.13.torch": "https://docs.pytorch.org/docs/2.13/generated/torch.",
  "pytorch-2.13.torch.nn":
    "https://docs.pytorch.org/docs/2.13/generated/torch.nn.",
  "pytorch-2.13.torch.nn.utils":
    "https://docs.pytorch.org/docs/2.13/generated/torch.nn.utils.",
  "sklearn.preprocessing":
    "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.",
};
//
// Here we write a simple switch-case statement to automatically choose the
// correct color scheme for the lecture materials.
//
var electroScience = [
  "Drive Technology",
  "Drive Systems",
  "Non-Linear Electronics",
];

var fundamentalScience = [
  "Higher Mathematics I",
  "Higher Mathematics II",
  "Electrodynamics",
];

var aiScience = ["Data Science I", "Data Science II"];

var roboticScience = ["Mobile Robotics", "Autonomous Mobile Robotics"];

var computerScience = ["Academic Writing"];

// Store the entry point to the website. This value needs to be kept as
// index.html.
var entryPoint = "index.html";

// Control how the margin images are checked. Start with false and make sure
// it is checked only once as it hangs the web-browser.
var bool_marginImageCheck = false;

// Used in representing what icons/links do and then can be relayed to modeline.
var ModeLineSelector = {};

// FUNCTIONS -------------------------------------------------------------------

/**
 * Detects if two elements are colliding
 *
 * Credit goes to BC on Stack Overflow, cleaned up a little bit
 *
 * @link http://stackoverflow.com/questions/5419134/how-to-detect-if-two-divs-touch-with-jquery
 * @param $div1
 * @param $div2
 * @returns {boolean}
 */
function util__isColliding($div1, $div2) {
  // Div 1 data
  var d1_offset = $div1.offset();
  var d1_height = $div1.outerHeight(true);
  var d1_width = $div1.outerWidth(true);
  var d1_distance_from_top = d1_offset.top + d1_height;
  var d1_distance_from_left = d1_offset.left + d1_width;

  // Div 2 data
  var d2_offset = $div2.offset();
  var d2_height = $div2.outerHeight(true);
  var d2_width = $div2.outerWidth(true);
  var d2_distance_from_top = d2_offset.top + d2_height;
  var d2_distance_from_left = d2_offset.left + d2_width;

  var not_colliding =
    d1_distance_from_top < d2_offset.top ||
    d1_offset.top > d2_distance_from_top ||
    d1_distance_from_left < d2_offset.left ||
    d1_offset.left > d2_distance_from_left;

  // Return whether it IS colliding
  return !not_colliding;
}

/**
 * Short-hand IF/ELSE to give a BOOLEAN if it is in given VIEW.
 *
 * @param view The view action is to be done, defined in `const_viewScreenGuide'
 * @return boolean value of the given VIEW.
 */
function util__describeViewState(view) {
  switch (view) {
    case "Max-Computer":
      if (
        window.matchMedia(
          "screen and (min-width: " +
            const_viewScreenGuide["Max-Computer"] +
            "px)",
        ).matches
      ) {
        return true;
      } else {
        return false;
      }

      break;

    case "Computer":
      if (
        window.matchMedia(
          "screen and (min-width: " + const_viewScreenGuide["Computer"] + "px)",
        ).matches
      ) {
        return true;
      } else {
        return false;
      }

      break;

    case "Tablet":
      if (
        window.matchMedia(
          "screen and (min-width: " + const_viewScreenGuide["Phone"] + "px)",
        ).matches &&
        window.matchMedia(
          "screen and (max-width: " + const_viewScreenGuide["Computer"] + "px)",
        ).matches
      ) {
        return true;
      } else {
        return false;
      }

      break;

    case "Phone":
      if (
        window.matchMedia(
          "screen and (max-width: " + const_viewScreenGuide["Phone"] + "px)",
        ).matches
      ) {
        return true;
      } else {
        return false;
      }

      break;
  }
}

/**
 * Allows insertion of text to the modeline.
 *
 * @param text Text to be shown in modeline
 *
 * @return
 */
function util__updateModeline(text) {
  $(".supp-line").text("");

  var newText = text;

  $(".supp-line").html("<p>" + newText + "</p>");
}

/**
 * Present the user the current view of WebBook.
 * The views supported are `Computer', `Tablet', and `Phone'.
 *
 * @return
 */
function menu__describeState() {
  // Here we do a small check-up and remove an unnecessary link from the menu
  if ($(".crosslinks-bottom").find("a").last().text().length == 0) {
    // $('.crosslinks-bottom').find('a').last().remove()
  }

  $(".crosslinks-bottom").find(".display-state").remove();

  if (util__describeViewState("Computer")) {
    $(".crosslinks-bottom").append(
      "<div class=display-state>[Computer-View]</div>",
    );
  } else if (util__describeViewState("Tablet")) {
    $(".crosslinks-bottom").append(
      "<div class=display-state>[Tablet-View]</div>",
    );
  }
}

function menu__getEqRef() {
  $("a").hover(
    function () {
      $(".eqref-block").remove();
      var $div = $("<div>", { class: "eqref-block" });
      $("article").append($div);
      link = $(this).attr("href").replace(/#/g, "");
      console.log(link);
      $div.append(
        $('[id="' + link + '"]')
          .parent()
          .clone(),
      );
      MathJax.typeset();
    },
    function () {
      $(".eqref-block").remove();
    },
  );
}

/**
 * Kills current tab. As we can not close it until unless, the tab was opened by
 * code we need to circumvent it by creating a faux tab.
 *
 *
 * @return
 */
function menu__killCurrentTab() {
  var win = window.open("about:blank", "_self");

  win.close();
}

function util__wrap(el, wrapper) {
  el.parentNode.insertBefore(wrapper, el);
  wrapper.appendChild(el);
}

function util__removeElementsByClass(className) {
  let elements = document.getElementsByClassName(className);
  while (elements.length > 0) {
    elements[0].parentNode.removeChild(elements[0]);
  }
}

function menu__makeCurrentPageTOC() {
  // Start by parsing the entire Chapter to get the content.

  util__removeElementsByClass("chapter-contents");

  const headings = document.querySelectorAll(
    "page h3, page h4, page h5, page h6",
  );
  const post = document.querySelector(".core-line");

  // Let's define the Page Contents in a state if the display is not
  // Max-Computer, it should show itself like a definition block (i.e.,
  // def-block) and if the display is full view, then keep is as a standard
  // div
  if (!util__describeViewState("Max-Computer")) {
    var details = document.createElement("dl");
    details.setAttribute("class", "chapter-contents def-block");

    var content = document.createElement("dt");
    content.setAttribute("class", "deffn def-line");
    content.setAttribute("id", "page-contents");

    var titleText = document.createElement("span");
    titleText.setAttribute("class", "category-def");
    titleText.innerHTML = "Page Contents";

    content.append(titleText);

    var pageLink = document.createElement("a");
    pageLink.setAttribute("class", "copiable-link");
    pageLink.setAttribute("href", "#page-contents");
    pageLink.innerHTML = "¶";

    details.prepend(pageLink);

    var inner = document.createElement("dd");
    var button = document.createElement("input");
    button.setAttribute("class", "body-button");
    button.setAttribute("type", "button");
    button.setAttribute("value", "Show/Hide");

    content.append(button);
  } else {
    var details = document.createElement("div");

    details.setAttribute("class", "chapter-contents");

    var content = document.createElement("div");
    content.setAttribute("class", "content");
    var inner = document.createElement("div");

    var summary = document.createElement("h3");
    summary.setAttribute("class", "header");
    summary.innerHTML = "Page Table of Contents";
  }

  inner.setAttribute("class", "inner");
  details.append(content);
  // content.append(summary);
  details.append(inner);

  var toc = document.createElement("div");
  toc.setAttribute("class", "toc");

  inner.append(toc);

  headings.forEach((el) => {
    const a = document.createElement("a");
    const title = document.createElement("span");
    title.setAttribute("class", "title");

    const number = document.createElement("span");
    number.setAttribute("class", "number");

    const header = document.createElement("span");

    // Remove any newlines from the string prior to parsing
    pre_header = el.textContent.replace(/[\r\n]+/gm, " ");

    if (el.tagName == "H3") {
      header.setAttribute("class", "sectionToc");

      a.textContent = pre_header.replace(/^\s*[0-9]+\.[0-9]+\s*(.*)/g, "$1");

      if (/^\s*([0-9]+\.[0-9]+).*/.test(pre_header)) {
        number.textContent = pre_header.replace(
          /^\s*([0-9]+\.[0-9]+).*/g,
          "$1",
        );
      } else {
        number.textContent = "";
      }
    } else if (el.tagName == "H4") {
      header.setAttribute("class", "subsectionToc");
      a.textContent = pre_header.replace(
        /^\s*[0-9]+\.[0-9]+\.[0-9]+\s*(.*)/g,
        "$1",
      );

      if (/^\s*([0-9]+\.[0-9]+\.[0-9]+).*/.test(pre_header)) {
        number.textContent = pre_header.replace(
          /^\s*([0-9]+\.[0-9]+\.[0-9]+).*/g,
          "$1",
        );
      } else {
        number.textContent = "";
      }
    } else {
    }

    title.append(a);
    pp = el.textContent.replace(
      /^\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s.*/g,
      "$1",
    );
    a.href = "#" + el.id;
    header.append(number);
    header.append(title);
    toc.append(header);
  });

  post.prepend(details);

  let defline = document.querySelector(".def-line");

  if (defline != null && defline.length > 1) {
    document
      .querySelector(".chapter-contents .inner")
      .append('<div class="code-toc"></div>');

    $(".def-block:not(.chapter-contents)").each(function () {
      // First determine the title of the block
      var _commandType = $(this)
        .find(".category-def")
        .text()
        .replace(/[\r\n]+/gm, " ");

      var _commandName = $(this)
        .find(".def-name")
        .text()
        .replace(/[\r\n]+/gm, " ");

      var _commandHeader =
        '<span class="title"><b>' +
        _commandType +
        '</b><span class="def-name">' +
        _commandName +
        "</span></span>";

      var _commandIDlink = $(this).find("a").attr("href");

      console.log(_commandIDlink);

      $(".chapter-contents")
        .find(".code-toc")
        .append(
          '<span class="generic-command-header">' +
            '<a href="' +
            _commandIDlink +
            '">' +
            _commandHeader +
            "</a>" +
            "</span><br>",
        );
    });
  }

  if ($(".exer-block").length > 0) {
    $(".chapter-contents")
      .find(".inner")
      .append(
        '<div class="toc-exer-contents">' +
          '<div class="section-header">' +
          '<h3 class="header">' +
          "Exercise Blocks" +
          "</h3>" +
          "</div>" +
          "</div>",
      );

    $(".exer-block").each(function () {
      // First determine the title of the block
      var _infoHeader = $(this)
        .find(".exer-upper")
        .text()
        .replace(
          /^\s*Example\s(.*?)\s*:\s*(.*)\s*$/g,
          '<span class="title">$2</span><span class="number">$1</span>',
        );

      // Generate the ID Link
      var _infoIDlink = $(this)
        .find(".exer-upper")
        .text()
        .replace(/^\s*Example\s(.*?)\s*:\s*(.*)\s*$/g, "exer-block $1")
        .replace(/ /g, "-")
        .replace(/\s*$/g, "");

      // Next add the ID attribute so it can be later directed.
      $(this).attr("id", _infoIDlink);

      $(".chapter-contents")
        .find(".toc-exer-contents")
        .append(
          '<p class="generic-info-header">' +
            '<a href="#' +
            _infoIDlink +
            '">' +
            _infoHeader +
            "</a>" +
            "</p>",
        );
    });
  }

  if ($(".theo-block").length > 0) {
    $(".chapter-contents")
      .find(".inner")
      .append(
        '<hr><div class="toc-contents block-contents toc-theo-contents"></div>',
      );

    $(".theo-block").each(function () {
      // First determine the title of the block
      var _infoHeader = $(this)
        .find(".theo-upper")
        .text()
        .replace(
          /^\s*Theory\s(.*?)\s*:\s*(.*)\s*$/g,
          '<span class="title">$2</span><span class="number">$1</span>',
        );

      // Generate the ID Link
      var _infoIDlink = $(this)
        .find(".theo-upper")
        .text()
        .replace(/^\s*Theory\s(.*?)\s*:\s*(.*)\s*$/g, "theo-block $1")
        .replace(/ /g, "-")
        .replace(/\s*$/g, "");

      // Next add the ID attribute so it can be later directed.
      $(this).attr("id", _infoIDlink);

      $(".chapter-contents")
        .find(".toc-theo-contents")
        .append(
          '<p class="generic-info-header">' +
            '<a href="#' +
            _infoIDlink +
            '">' +
            _infoHeader +
            "</a>" +
            "</p>",
        );
    });
  }

  if ($(".info-block").length > 0) {
    $(".chapter-contents")
      .find(".inner")
      .append(
        '<hr><div class="toc-contents block-contents toc-info-contents"></div>',
      );

    $(".info-block").each(function () {
      // First determine the title of the block
      var _infoHeader = $(this)
        .find(".upper")
        .text()
        .replace(
          /^\s*Information\s(.*?)\s*:\s*(.*)\s*$/g,
          '<span class="title"><b>Information: </b>$2</span> <span class="number">$1</span>',
        );

      // Generate the ID Link
      var _infoIDlink = $(this)
        .find(".upper")
        .text()
        .replace(/^\s*Information\s(.*?)\s*:\s*(.*)\s*$/g, "info-block $1")
        .replace(/ /g, "-")
        .replace(/\s*$/g, "");

      // Next add the ID attribute so it can be later directed.
      $(this).attr("id", _infoIDlink);

      $(".chapter-contents")
        .find(".toc-info-contents")
        .append(
          '<p class="generic-info-header">' +
            '<a href="#' +
            _infoIDlink +
            '">' +
            _infoHeader +
            "</a>" +
            "</p>",
        );
    });
  }

  if (!util__describeViewState("Max-Computer")) {
    $(".chapter-contents").insertAfter(".crosslinks-top");
  }
  {
  }
}

/**
 * Refresh Page if content is stuck.
 *
 *
 * @return
 */
function menu__refreshPage() {
  window.location.reload();
}

function navigate__gotoNextChapter() {
  let hreflink = document
    .querySelector(".crosslinks-top")
    .children[0].querySelector("a")
    .getAttribute("href");

  if (typeof hreflink == "undefined") {
    window.alert(
      "Reached end of WebBook. There are no more chapters to continue...",
    );
  } else {
    window.location.href = hreflink;
  }
}

function navigate__gotoPrevChapter() {
  let hreflink = document
    .querySelector(".crosslinks-top")
    .children[1].querySelector("a")
    .getAttribute("href");

  if (typeof hreflink == "undefined") {
    window.alert(
      "Reached beginning of WebBook. There are no more chapters to go back...",
    );
  } else {
    window.location.href = hreflink;
  }
}

function navigate__gotoTopOfPage() {
  document.body.scrollTop = document.documentElement.scrollTop = 0;
}

function navigate__toBottomOfPage() {
  window.scrollTo(0, document.body.scrollHeight);
}

function navigate__gotoIndexPage() {
  if (document.URL.includes("index.html")) {
    window.alert("You are already on Entry Point.");
  } else {
    window.location.href = entryPoint;
  }
}

function navigate__gotoChapterPage() {
  // We check by looking at the documents 'h2' header and determine what to do
  // based on its value. If the values is `null' then we assume we are NOT ON
  // CHAPTER page.
  let h2 = document.querySelector("h2");

  if (h2 == null) {
    let hreflink = document
      .querySelector(".crosslinks-top")
      .children[2].querySelector("a")
      .getAttribute("href");

    window.location.href = hreflink;
  } else {
    window.alert("You are already on a Chapter page. Nowhere to go!");
  }
}

// Display Hiders --------------------------------------------------------------

// Here we create a simple function to hide exercises and other relevant boxes
// as they are not immediately relevant to the reader.

let reader_mode = false;

function view__reader() {
  if ($("nav.wide").first().css("top") == "50px") {
    $("nav.wide").animate({ top: "0" });
    $("nav.wide").animate({ height: "100vh" });
  } else {
    $("nav.wide").animate({ top: "50px" });
    $("nav.wide").animate({ height: "calc(100vh - 50px)" });
  }

  $(".mode-line").fadeToggle();
}

function ShowHideBlock() {
  let button = document.querySelectorAll(".body-button");

  for (const el of button) {
    el.addEventListener("click", () => {
      let showState = el.parentNode.nextElementSibling.style.display;

      if (showState != "block") {
        el.parentNode.nextElementSibling.style.display = "block";
      } else {
        el.parentNode.nextElementSibling.style.display = "none";
      }
    });
  }
}

// Code Higlighting ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", (event) => {
  document.querySelectorAll("pre code").forEach((block) => {
    hljs.addPlugin(new CopyButtonPlugin());
    hljs.highlightElement(block);
  });
});

/**
 * Determine which colour is to be used when presenting content. In the event a
 * domain cant be picked, default it to `--org-science'.
 *
 *
 * @return
 */
function filter__colourDomainPicker() {
  // Determine the name of the lecture/project to determine its domain
  var currentDomain = $(".header-lecture-name").text();

  switch (true) {
    case electroScience.includes(currentDomain):
      $(":root").css("--chosen-science", "var(--electro-science)");

      break;

    case fundamentalScience.includes(currentDomain):
      $(":root").css("--chosen-science", "var(--fundamental-science)");

      break;

    case roboticScience.includes(currentDomain):
      $(":root").css("--chosen-science", "var(--robotic-science)");

      break;

    case aiScience.includes(currentDomain):
      $(":root").css("--chosen-science", "var(--ai-science)");

      break;

    case computerScience.includes(currentDomain):
      $(":root").css("--chosen-science", "var(--computer-science)");

      break;

    default:
      $(":root").css("--chosen-science", "var(--org-science)");

      break;
  }
}

// Remove PDF Link on Chapter References page ----------------------------------

/**
 * Inserts alternating row colors. This is done as JS is used to remove empty
 * rows and this messes the coloring done by statics pages.
 *
 *
 * @return
 */
function alternatingRowColors() {
  $(document).ready(function () {
    $("tr:odd").css({
      "background-color": "var(--table-row-color)",
    });
  });
}

/**
 * Looks at the content folder to retrive the lecture structure page and
 * displays with proper formatting. This is bound to key for easy navigation.
 *
 *
 * @return
 */
function menu__showLectureStructure() {
  window.location.href = "./content/lecture-structure.html";
}

function menu__createDetailTOC() {
  //  $('article').append("<div class='reference-page'><div class='detail-toc'></nav></div>");

  $.ajax({
    url: "content/toc-detail.html",
    type: "GET",

    success: function (data) {
      $(".reference-page").find(".detail-toc").html($(data));
    },
  });
}

function menu__toggleDetailTOC() {
  // First determine the current state of the primary TOC.
  const otpState = $(".reference-page").css("display");

  if (otpState == "none") {
    $(".reference-page")
      .stop()
      .animate({ height: "toggle", opacity: "toggle" }, "slow");
  } else {
    $(".reference-page")
      .stop()
      .animate({ height: "toggle", opacity: "toggle" }, "slow");
  }
}

function menu__CreateIndexSearch() {
  $("html").append('<div class="index-search"></div>');

  $.ajax({
    url: "info_block.db",
    type: "GET",

    success: function (data) {
      $(".index-search").html($(data));
      document.querySelectorAll("pre code").forEach((block) => {
        hljs.addPlugin(new CopyButtonPlugin());
        hljs.highlightElement(block);
      });
    },
  });
}

/**
 * Fixes the problem where the citations within margin comments whould shoot to
 * the side of the margin. Now they would be put below the margin comment.
 *
 *
 * @return
 */
function FixSideMarginDerivatives() {
  $(".sidenote").find(".cite-detail").css({
    position: "fixed",
    top: "unset",
    bottom: "50px",
    right: "20px",
  });
}

/**
 * Looks the `code' in documentation to retrieve the necessary documentation and
 * present the user with the information on the page
 *
 * @param string to be searched.
 *
 * @return the website in a window
 */
function FindReference(code) {
  $("article").append("<div class='reference-page'></div>");

  $.each(DocumentationSites, function (index, element) {
    $.ajax({
      url: DocumentationSites[index] + code + ".html",
      type: "GET",
      success: function (data) {
        // $('.documentation').html($(data).find('section').html());

        $(".reference-page").append(
          '<iframe src="' +
            DocumentationSites[index] +
            code +
            ".html" +
            '" title="description"></iframe>',
        );
      },
      error: function (data) {},
    });
  });
}

function menu__VisitCorg() {
  /* Using the header, find the name of the .corg file */
  var corg_name = $(".header-chapter").text();

  /* Then combine the strings to generate the valid HTML */
  var hreflink = "content/corg/" + corg_name + "s.html";

  $.ajax({
    url: hreflink,
    type: "GET",
    success: function (data) {
      window.location.href = hreflink;
    },
    error: function (data) {
      window.alert(
        "This page either has no related code NoteBook, or you are already on one.",
      );
    },
  });
}

function filter__removeLineswithinFloats() {
  var TableFilterListClass = [".threeparttable", ".float"];

  for (const element of TableFilterListClass) {
    $(element)
      .contents()
      .filter(function () {
        return this.nodeType === 3 && this.nodeValue.trim() !== "";
      })
      .remove();
  }
}

function filter__repositionCaptioninMargin() {
  $(".sidenote")
    .find(".figure")
    .find(".caption")
    .each(function () {
      $(this).appendTo($(this).parent());
    });
}

function filter__removeUnnecessaryTOCs() {
  $(".chapter").find(".subsectionTOCS").remove();
}

function filter__readjustMarginImages() {
  var marginImages = $(".sidenote").find(".sidenote-image").parent();
  var sideImages = $(".sidenote").find(".figure").parent();

  var allMarginImages = marginImages.add(sideImages);

  var collidecheck = [false];

  while (collidecheck.includes(false)) {
    var collidecheck = [];

    for (var i = 0; i < allMarginImages.length - 1; i++) {
      // First we determine which elements are touching.
      if (
        util__isColliding(allMarginImages.eq(i), allMarginImages.eq(i + 1)) ==
        true
      ) {
        var _tmp = parseInt(allMarginImages[i + 1].offsetTop, 10);

        allMarginImages[i + 1].style.top = _tmp + 40 + "px";

        var marginImages = $(".sidenote").find(".sidenote-image").parent();
        var sideImages = $(".sidenote").find(".figure").parent();

        var allMarginImages = marginImages.add(sideImages);

        collidecheck.push(false);
      } else {
        collidecheck.push(true);
      }
    }
  }
}

function set__Theme(state) {
  let code_sheet = document.getElementById("code-style");

  // Control the color settings of the page
  if (state == true) {
    document.body.classList.toggle("dark-mode");
    code_sheet.setAttribute("href", "style-code/gruvbox-dark-hard.min.css");
  } else {
    document.body.classList.remove("dark-mode");
    code_sheet.setAttribute("href", "style-code/gruvbox-light-hard.min.css");
  }
}

function menu__switchTheme() {
  let darkMode = localStorage.getItem("darkMode") == "true";

  darkMode = !darkMode;

  set__Theme(darkMode);

  window.localStorage.setItem("darkMode", darkMode); //save it in local storage
}

function anim__highlightSideNoteImage() {
  $(".margin, .margin-comment, .acronym, .cite").hover(
    function () {
      // We make sure to first make all other `.sidenote' to have
      // transparency.
      $(".margin")
        .not(this)
        .find(".sidenote")
        .stop()
        .animate({ opacity: "0.15" });
      $(".sidenote").find(".figure").stop().animate({ opacity: "0.15" });
    },
    function () {
      $(".margin")
        .not(this)
        .find(".sidenote")
        .stop()
        .animate({ opacity: "1.0" });
      $(".sidenote").find(".figure").stop().animate({ opacity: "1.0" });
    },
  );
}

function modeline__showLinkInformation() {
  for (var key in ModeLineSelector) {
    let value = ModeLineSelector[key];

    $(key).hover(
      function () {
        util__updateModeline(value);
      },
      function () {
        util__updateModeline("");
      },
    );
  }
}

function menu__toggleTOC() {
  let navWide = document.querySelector("nav.wide");

  // First determine the current state of the primary TOC.
  const tocState = getComputedStyle(navWide).display;

  $(".chapter").attr("id", "blur-transition");

  if (tocState == "none") {
    $("nav.wide")
      .stop()
      .animate({ width: "toggle", opacity: "toggle" }, "slow");
    $(".crosslinks-bottom").css("width", "100%");

    // Show screen size specific actions
    if (util__describeViewState("Computer")) {
    } else if (util__describeViewState("Tablet")) {
      // add blur to the background when the table of contents is shown.
      $(".crosslinks-bottom").hide();
      $(".chapter-contents").hide();
      $(".chapter").addClass("blur");
    }
  } else {
    $("nav.wide")
      .stop()
      .animate({ width: "toggle", opacity: "toggle" }, "slow");
    $(".crosslinks-bottom").animate({ width: "100%" });

    if (util__describeViewState("Computer")) {
    } else if (util__describeViewState("Tablet")) {
      // add blur to the background when the table of contents is shown.
      $(".crosslinks-bottom").animate(
        { height: "toggle", opacity: "toggle" },
        "slow",
      );

      $(".chapter").removeClass("blur");
    }
  }
}

/**
 * Enable/disable the viewing of the current page Table of contents.
 *
 *
 * @return
 */
function menu__toggleCurrentPageTOC() {
  // First determine the current state of the primary TOC.
  const otpState = $(".chapter-contents").css("display");

  if (otpState == "none") {
    $(".chapter-contents")
      .stop()
      .animate({ width: "toggle", opacity: "toggle" }, "slow");
  } else {
    $(".chapter-contents")
      .stop()
      .animate({ width: "toggle", opacity: "toggle" }, "slow");
  }
}

function modeline__toggleThemeButton() {
  var themeToggleButton = `<button type="button"
class="toggle-button theme-toggle" onclick="menu__switchTheme()" title="Switch Themes">
<i class="fa fa-paint-brush"></i></button>`;

  var tocToggleButton = `<button type="button"
class="toggle-button toc-toggle" onclick="menu__toggleTOC()" title="Toggle Table of contents">
<i class="fa fa-bars"></i></button>`;

  var themeGroup = document.createElement("span");

  themeGroup.classList.add("small-button");

  themeGroup.innerHTML = tocToggleButton + themeToggleButton;

  var parentNode = document.querySelector(".button-group");

  parentNode.appendChild(themeGroup);
}

function modeline__generateModeMenuList(json) {
  // First retrieve the first depth list which will store the important
  // elements
  var mainMenu = Object.keys(moodMenu);

  // We then create the primary menu which will be displate in the mode-line.
  for (var i = 0; i < mainMenu.length; i++) {
    $(".button-group").append(
      '<div class="dropdown">' +
        '<button class="dropbtn" id="btn-' +
        mainMenu[i] +
        '">' +
        mainMenu[i] +
        "</button>" +
        '<div class="dropdown-content">' +
        "</div>" +
        "</div>",
    );

    var subList = Object.keys(moodMenu[Object.keys(moodMenu)[i]][0]);

    for (var ii = 0; ii < subList.length; ii++) {
      if (subList[ii].includes("Sep-")) {
        $(".button-group")
          .find("#btn-" + mainMenu[i])
          .next(".dropdown-content")
          .append("<hr>");
      } else {
        let keyboard =
          moodMenu[Object.keys(moodMenu)[i]][0][subList[ii]][0].key;

        if (keyboard == "nil") {
          kbd = "";
        } else {
          kbd = "<kbd>" + keyboard + "</kbd>";
        }

        $(".button-group")
          .find("#btn-" + mainMenu[i])
          .next(".dropdown-content")
          .append(
            '<span class="menu-element">' +
              '<a onclick="' +
              moodMenu[Object.keys(moodMenu)[i]][0][subList[ii]][0].command +
              '">' +
              subList[ii] +
              "</a>" +
              kbd +
              "<br>",
          );
      }
    }
  }
}

document.addEventListener("DOMContentLoaded", function () {
  let darkMode = localStorage.getItem("darkMode") == "true";

  menu__makeCurrentPageTOC();

  //$.when(removeEmptyRows()).then(alternatingRowColors());

  FixSideMarginDerivatives();

  modeline__showLinkInformation();

  filter__removeLineswithinFloats();

  filter__repositionCaptioninMargin();

  set__Theme(darkMode);

  filter__removeUnnecessaryTOCs();

  modeline__generateModeMenuList(moodMenu);

  // anim__highlightSideNoteImage ();

  menu__createDetailTOC();

  modeline__toggleThemeButton();
});

// RESIZING FUNCTIONS ----------------------------------------------------------

$(window).resize(function () {
  menu__makeCurrentPageTOC();

  ShowHideBlock();

  // Update the user on which view mode they are viewing their content on
  menu__describeState();

  if (
    util__describeViewState("Max-Computer") &&
    bool_marginImageCheck == false
  ) {
    filter__readjustMarginImages();
    bool_marginImageCheck = true;
  }
});

// DOM-LOADED FUNCTIONS --------------------------------------------------------

$(document).ready(function () {
  // Determine the colour to be used for the topic.
  filter__colourDomainPicker();

  // Update the user on which view mode they are viewing their content on
  menu__describeState();

  // Remove the TOC if it is displayed in an unsupported view

  ShowHideBlock();

  if (
    util__describeViewState("Max-Computer") &&
    bool_marginImageCheck == false
  ) {
    filter__readjustMarginImages();
    bool_marginImageCheck = true;
  }
});

// Keyboard Event Control ------------------------------------------------------

// The following keyboard control add numerous keyboard shortcuts to useful
// functions for easy navigation and quick information searching. We need to
// make sure `pagefind' window is not currently active as it causes problem with
// navigation.

document.addEventListener("keypress", function (event) {
  if (document.activeElement.tagName == "BODY") {
    // We just need to make sure these events only occur when BODY is the
    // active element and not when the pagefind is active.
    switch (event.key) {
      case "w":
        view__reader();

        break;

      case "n":
        navigate__gotoNextChapter();
        break;

      case "p":
        navigate__gotoPrevChapter();
        break;

      case "b":
        window.scrollTo(0, document.body.scrollHeight);
        break;

      case "u":
        document.body.scrollTop = document.documentElement.scrollTop = 0;
        break;

      case "d":
        navigate__gotoChapterPage();
        break;

      case "l":
        menu__showLectureStructure();
        break;

      case ",":
        menu__toggleDetailTOC();
        break;

      case ".":
        if (util__describeViewState("Max-Computer")) {
          menu__toggleCurrentPageTOC();
        }
        break;

      case "i":
        window.location.href = entryPoint;
        break;

      case "r":
        menu__refreshPage();
        break;
    }
  }
});

// -----------------------------------------------------------------------------
// flatjs.js ends here.
//
//
