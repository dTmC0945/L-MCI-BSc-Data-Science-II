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

// Here we store the names of the tables we want them to be treated like
// DataStructures.
var DataTableList = ["TOPIC-NAMES"];

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

            "Bookmark This Page": [
                {
                    key: "nil",
                    command: "menu__bookmarkThisPage()",
                },
            ],

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

// -----------------------------------------------------------------------------

/**
 * Add a Copy button to the code snipped along with copy to clipboard
 * functionality.
 *
 * @return
 */
function button_codeCopy() {
    // Find the code header
    let codeBlocks = document.getElementsByClassName("code-header");

    // Define the two types of icons to be used.
    // iconCopy  -- used in showing whether it is copiable
    // iconCheck -- used in showing the icon was clicked
    let iconCopy = '<i class="fa-regular fa-clipboard"></i>';
    let iconCheck = '<i class="fa-regular fa-circle-check"></i>';

    for (let i = 0; i < codeBlocks.length; i++) {
        // Here we create the button and add the necessary customisation.
        let codeHeader = codeBlocks.item(i);
        let button = document.createElement("button");

        button.setAttribute("class", "copy-button");
        button.innerHTML = iconCopy;
        codeHeader.prepend(button);

        let textClicked = document.createElement("span");
        textClicked.setAttribute('class', 'copy-text');
        textClicked.innerText = "Copied!";

        codeHeader.append(textClicked)

        // Next we write a fuction whereby we look where the button is and based
        // on its position we try to find the code snippet. Once found we then
        // retrieve the code
        button.addEventListener(
            "click",
            function (btn) {
                let codeSnippet =
                    btn.target.closest("div").nextElementSibling.innerText;

                navigator.clipboard.writeText(codeSnippet);
                button.innerHTML = iconCheck;
                textClicked.setAttribute('class', 'copy-text-is-visible');
                                
                setTimeout(function () {
                    // After a second has passed, revert to the original icon.
                    button.innerHTML = iconCopy;
                    textClicked.setAttribute('class', 'copy-text');
                }, 1000);
            },
            false,
        );
    }
}

// -----------------------------------------------------------------------------

function menu__makeCurrentPageTOC() {
    // Start by parsing the entire Chapter to get the content.

    util__removeElementsByClass("chapter-contents");

    const headings = document.querySelectorAll(
        "article h3, article h4, article h5",
    );
    const post = document.querySelector(".core-line");

    // Let's define the Page Contents in a state if the display is not
    // Max-Computer, it should show itself like a definition block (i.e.,
    // def-block) and if the display is full view, then keep is as a standard
    // div

    var details = document.createElement("div");

    details.setAttribute("class", "chapter-contents");

    var content = document.createElement("div");
    content.setAttribute("class", "content");
    var inner = document.createElement("div");

    var summary = document.createElement("h3");
    summary.setAttribute("class", "header");
    summary.innerHTML = "Page Table of Contents";

    inner.setAttribute("class", "inner");
    details.append(content);
    // content.append(summary);
    details.append(inner);

    var toc = document.createElement("div");
    toc.setAttribute("class", "toc");

    inner.append(toc);

    var logoButtons = document.createElement("div");
    logoButtons.setAttribute("class", "button-logo");

    toc.append(logoButtons);

    // Here we loop over all the defined headers in the page and then write down
    // their links. We look for the `title' and the `number' classes and get
    // their content.
    headings.forEach((el) => {
        // Let's disregard some elements as they take unnecessary space on the
        // list.
        if (
            el.innerText == "Page References" ||
                el.innerText == "Page Glossary Terms" ||
                el.innerText == "Page Footnotes"
        ) {
            let pageDetails = document.createElement("a");
            let logo_pageDetails = document.createElement("i");
            headerlink = el.querySelector("a").id;
            logo_pageDetails.setAttribute("class", "fa-solid fa-magnifying-glass");
            pageDetails.setAttribute("href", "#" + headerlink);
            pageDetails.setAttribute("title", el.innerText);
            pageDetails.append(logo_pageDetails);
            logoButtons.append(pageDetails);

            return;
        }

        const a = document.createElement("a");
        const title = document.createElement("span");
        title.setAttribute("class", "title");
        const number = document.createElement("span");
        number.setAttribute("class", "number");

        const header = document.createElement("span");

        // Check each state of header and assign their specific class
        switch (el.tagName) {
        case "H2":
            header.setAttribute("class", "chapterToc");
            break;
        case "H3":
            header.setAttribute("class", "sectionToc");
            break;
        case "H4":
            header.setAttribute("class", "subsectionToc");
            break;
        case "H5":
            header.setAttribute("class", "subsectionToc");
            break;
        }

        a.textContent = el.querySelector(".title").textContent;
        number.textContent = el.querySelector(".number").textContent;

        title.append(a);
        a.href = "#" + el.id;
        header.append(number);
        header.append(title);
        toc.append(header);
    });

    post.prepend(details);

    let defBlock = document.querySelector(".def-line");

    if (defBlock != null && defBlock.length > 1) {
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

function ShowHideBlock(button) {
    
    for (const el of button) {

        let blockBody = el.parentNode.nextElementSibling;
        
        blockBody.setAttribute('class', 'dd-block');
        
        el.addEventListener("click", () => {        

            if (blockBody.className == "dd-block") {
                blockBody.setAttribute('class', 'dd-block-is-visible');
                
            } else {
                blockBody.setAttribute('class', 'dd-block');
            }
        });
    }
}

// Code Higlighting ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", (event) => {
    document.querySelectorAll("pre code").forEach((block) => {
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

function menu__bookmarkThisPage() {
    if (window.sidebar && window.sidebar.addPanel) {
        // Mozilla Firefox Bookmark
        window.sidebar.addPanel(document.title, window.location.href, "");
    } else if (window.external && "AddFavorite" in window.external) {
        // IE Favorite
        window.external.AddFavorite(location.href, document.title);
    } else if (window.opera && window.print) {
        // Opera Hotlist
        this.title = document.title;
    } else {
        // webkit - safari/chrome
        alert(
            "Press " +
                (navigator.userAgent.toLowerCase().indexOf("mac") != -1
                 ? "Command/Cmd"
                 : "CTRL") +
                " + D to bookmark this page.",
        );
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
            .animate({ width: "toggle", opacity: "toggle" }, "slow", function () {
                nav__ensureActiveVisible(navWide);
            });

        // Show screen size specific actions
        if (util__describeViewState("Tablet") || util__describeViewState("Phone")) {
            // add blur to the background when the table of contents is shown.
            $(".chapter-contents").hide();
            $(".chapter").addClass("blur");
        }
    } else {
        $("nav.wide")
            .stop()
            .animate({ width: "toggle", opacity: "toggle" }, "slow");

        if (util__describeViewState("Tablet") || util__describeViewState("Phone")) {
            // add blur to the background when the table of contents is shown.
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

    var pageTOCToggleButton = `<button type="button"
class="toggle-button pagetoc-toggle" onclick="menu__toggleCurrentPageTOC()" title="Toggle Page Table of Contents">
<i class="fa fa-square-minus"></i></button>`;

    var tocToggleButton = `<button type="button"
class="toggle-button toc-toggle" onclick="menu__toggleTOC()" title="Toggle Table of contents">
<i class="fa fa-bars"></i></button>`;

    var themeGroup = document.createElement("span");

    themeGroup.classList.add("small-button");

    themeGroup.innerHTML =
        tocToggleButton + pageTOCToggleButton + themeToggleButton;

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

// NAVBAR CONFIGURATIONS -------------------------------------------------------

// Here we add suggested improvements to the side navigation bar to improve user
// interaction.

// Let's start with storing the session variables.
const TOC_SCROLL_STORAGE_KEY = "webbook-toc-scrollTop";
const TOC_OPEN_CHAPTERS_STORAGE_KEY = "webbook-toc-open-chapters";

/**
 * Retrieves the stored key in the browser.
 *
 * @param key : key to be retrieved from the web browser.
 *
 * @return
 */
function nav__storageGet(key) {
    try {
        return window.sessionStorage.getItem(key);
    } catch (e) {
        return null;
    }
}

/**
 * Store a given value to the browser.
 *
 * @param key   : the variable to be stored.
 * @param value : the value of the variable to be stored.
 *
 * @return
 */
function nav__storageSet(key, value) {
    try {
        window.sessionStorage.setItem(key, value);
    } catch (e) {
        // Ignore storage errors (e.g. restricted browsing contexts)
    }
}

/**
 * Determine which chapter is CURRENTLY active from the stored
 * variable `TOC_OPEN_CHAPTERS_STORAGE_KEY'.
 *
 * @return
 */
function nav__getOpenChaptersMap() {
    const raw = nav__storageGet(TOC_OPEN_CHAPTERS_STORAGE_KEY);
    if (!raw) {
        return {};
    }
    try {
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch (e) {
        return {};
    }
}

function nav__saveChapterOpenState(chapterKey, isOpen) {
    if (!chapterKey) {
        return;
    }
    const map = nav__getOpenChaptersMap();
    map[chapterKey] = Boolean(isOpen);
    nav__storageSet(TOC_OPEN_CHAPTERS_STORAGE_KEY, JSON.stringify(map));
}

function nav__setChapterGroupOpen(group, isOpen) {
    if (!group || !group.classList.contains("toc-has-children")) {
        return;
    }
    group.classList.toggle("toc-collapsed", !isOpen);
    const toggleBtn = group.querySelector(
        ":scope > .chapterToc .toc-chapter-toggle," +
            ":scope > .appendixToc .toc-chapter-toggle",
    );
    if (toggleBtn) {
        toggleBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
    }
}

function nav__normalizePathname(pathname) {
    if (!pathname) {
        return "/index.html";
    }
    return pathname.endsWith("/") ? pathname + "index.html" : pathname;
}

function nav__isLinkForCurrentPage(anchor) {
    if (!anchor) {
        return false;
    }
    const rawHref = (anchor.getAttribute("href") || "").trim();
    if (!rawHref) {
        return false;
    }
    if (rawHref.startsWith("#")) {
        return true;
    }
    try {
        const linkUrl = new URL(rawHref, window.location.href);
        return (
            linkUrl.origin === window.location.origin &&
                nav__normalizePathname(linkUrl.pathname) ===
                nav__normalizePathname(window.location.pathname)
        );
    } catch (e) {
        return false;
    }
}

function nav__buildCollapsibleTOC(tocEl) {
    const chapterSelector = "span.chapterToc, span.appendixToc";
    const childSelector =
          "span.sectionToc, span.subsectionToc, span.subsubsectionToc";

    const nodes = Array.from(tocEl.children);
    let currentGroup = null;
    let currentChildren = null;

    nodes.forEach((node) => {
        if (node.matches(chapterSelector)) {
            currentGroup = document.createElement("div");
            currentGroup.className = "toc-chapter-group";

            const numberText = (
                node.querySelector(".number")?.textContent || ""
            ).trim();
            const titleText = (node.querySelector(".title")?.textContent || "")
                  .replace(/\s+/g, " ")
                  .trim();
            currentGroup.dataset.chapterKey = numberText + ":" + titleText;

            currentChildren = document.createElement("div");
            currentChildren.className = "toc-chapter-children";

            tocEl.insertBefore(currentGroup, node);
            currentGroup.appendChild(node);
            currentGroup.appendChild(currentChildren);
        } else if (node.matches(childSelector) && currentChildren) {
            currentChildren.appendChild(node);
        } else {
            currentGroup = null;
            currentChildren = null;
        }
    });

    tocEl.querySelectorAll(".toc-chapter-group").forEach((group) => {
        const childrenWrap = group.querySelector(":scope > .toc-chapter-children");
        const chapterHeader = group.querySelector(":scope > " + chapterSelector);

        if (!childrenWrap || childrenWrap.children.length === 0 || !chapterHeader) {
            if (childrenWrap) {
                childrenWrap.remove();
            }
            return;
        }

        group.classList.add("toc-has-children");

        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "toc-chapter-toggle";
        toggleBtn.setAttribute("aria-expanded", "false");
        toggleBtn.setAttribute("aria-label", "Toggle chapter sections");
        toggleBtn.innerHTML =
            '<i class="fa-solid fa-chevron-right" aria-hidden="true"></i>';

        toggleBtn.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            const willOpen = group.classList.contains("toc-collapsed");
            nav__setChapterGroupOpen(group, willOpen);

            if (tocEl.dataset.filtering !== "true") {
                nav__saveChapterOpenState(group.dataset.chapterKey, willOpen);
            }
        });

        chapterHeader.insertBefore(toggleBtn, chapterHeader.firstChild);
    });
}

function nav__applySavedChapterStates(tocEl) {
    const savedMap = nav__getOpenChaptersMap();

    tocEl
        .querySelectorAll(".toc-chapter-group.toc-has-children")
        .forEach((group) => {
            const containsCurrentPage = Boolean(
                group.querySelector(".toc-current-page"),
            );
            const key = group.dataset.chapterKey;
            const isOpen = containsCurrentPage || savedMap[key] === true;
            nav__setChapterGroupOpen(group, isOpen);
        });
}

function nav__buildStickySearch(navWide, tocEl) {
    const stickyBar = document.createElement("div");
    stickyBar.className = "toc-sticky-bar";

    const pagefindTrigger = tocEl.querySelector("pagefind-modal-trigger");
    if (pagefindTrigger) {
        stickyBar.appendChild(pagefindTrigger);
    }

    const filterWrap = document.createElement("div");
    filterWrap.className = "toc-filter-wrap";

    const icon = document.createElement("i");
    icon.className = "fa-solid fa-magnifying-glass toc-filter-icon";
    icon.setAttribute("aria-hidden", "true");

    const input = document.createElement("input");
    input.type = "search";
    input.className = "toc-filter-input";
    input.placeholder = "Filter headers...";
    input.setAttribute("aria-label", "Filter table of contents");
    input.autocomplete = "off";
    input.spellcheck = false;

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "toc-filter-clear";
    clearBtn.setAttribute("aria-label", "Clear filter");
    clearBtn.title = "Clear filter";
    clearBtn.innerHTML = "&times;";
    clearBtn.hidden = true;

    filterWrap.appendChild(icon);
    filterWrap.appendChild(input);
    filterWrap.appendChild(clearBtn);
    stickyBar.appendChild(filterWrap);

    navWide.insertBefore(stickyBar, tocEl);

    const emptyMsg = document.createElement("div");
    emptyMsg.className = "toc-filter-empty";
    emptyMsg.textContent = "No matching sections";
    emptyMsg.hidden = true;

    const navFooter = tocEl.querySelector(".nav-footer");
    if (navFooter) {
        tocEl.insertBefore(emptyMsg, navFooter);
    } else {
        tocEl.appendChild(emptyMsg);
    }

    const getNormalizedText = (el) =>
          (el ? el.textContent : "").replace(/\s+/g, " ").trim().toLowerCase();

    const applyFilter = (rawQuery) => {
        const query = rawQuery.trim().toLowerCase();
        clearBtn.hidden = query.length === 0;

        const groups = tocEl.querySelectorAll(".toc-chapter-group");

        if (!query) {
            delete tocEl.dataset.filtering;
            groups.forEach((group) => {
                group.classList.remove("toc-filter-hidden");
                group
                    .querySelectorAll(".toc-filter-hidden")
                    .forEach((el) => el.classList.remove("toc-filter-hidden"));
            });
            emptyMsg.hidden = true;
            nav__applySavedChapterStates(tocEl);
            return;
        }

        tocEl.dataset.filtering = "true";
        let anyVisible = false;

        groups.forEach((group) => {
            const chapterHeader = group.querySelector(
                ":scope > span.chapterToc, :scope > span.appendixToc",
            );
            const childrenWrap = group.querySelector(
                ":scope > .toc-chapter-children",
            );
            const childItems = childrenWrap ? Array.from(childrenWrap.children) : [];

            const chapterMatches = getNormalizedText(chapterHeader).includes(query);

            if (chapterMatches) {
                group.classList.remove("toc-filter-hidden");
                childItems.forEach((item) =>
                    item.classList.remove("toc-filter-hidden"),
                );
                nav__setChapterGroupOpen(group, true);
                anyVisible = true;
                return;
            }

            let hasMatchingChild = false;
            let currentSection = null;
            let currentSectionMatched = false;

            childItems.forEach((item) => {
                const matches = getNormalizedText(item).includes(query);
                if (item.classList.contains("sectionToc")) {
                    currentSection = item;
                    currentSectionMatched = matches;
                    item.classList.toggle("toc-filter-hidden", !matches);
                    if (matches) {
                        hasMatchingChild = true;
                    }
                } else {
                    const showItem = matches || currentSectionMatched;
                    item.classList.toggle("toc-filter-hidden", !showItem);
                    if (matches) {
                        hasMatchingChild = true;
                        if (currentSection) {
                            currentSection.classList.remove("toc-filter-hidden");
                        }
                    }
                }
            });

            if (hasMatchingChild) {
                group.classList.remove("toc-filter-hidden");
                nav__setChapterGroupOpen(group, true);
                anyVisible = true;
            } else {
                group.classList.add("toc-filter-hidden");
            }
        });

        emptyMsg.hidden = anyVisible;
    };

    input.addEventListener("input", function () {
        applyFilter(input.value);
    });

    input.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            event.preventDefault();
            input.value = "";
            applyFilter("");
            input.blur();
        }
    });

    input.addEventListener("keypress", function (event) {
        event.stopPropagation();
    });

    clearBtn.addEventListener("click", function () {
        input.value = "";
        applyFilter("");
        input.focus();
    });
}

function nav__saveTOCScroll(navWide) {
    if (!navWide) {
        return;
    }
    nav__storageSet(
        TOC_SCROLL_STORAGE_KEY,
        String(Math.round(navWide.scrollTop)),
    );
}

function nav__ensureActiveVisible(navWide) {
    if (!navWide) {
        return;
    }
    if (
        getComputedStyle(navWide).display === "none" ||
            navWide.clientHeight === 0
    ) {
        return;
    }

    const activeEl = navWide.querySelector(".toc .toc-active");
    if (!activeEl || activeEl.offsetParent === null) {
        return;
    }

    const navRect = navWide.getBoundingClientRect();
    const stickyBar = navWide.querySelector(".toc-sticky-bar");
    const stickyBottom = stickyBar
          ? Math.max(navRect.top, stickyBar.getBoundingClientRect().bottom)
          : navRect.top;
    const activeRect = activeEl.getBoundingClientRect();
    const buffer = 8;

    const isVisible =
          activeRect.top >= stickyBottom + buffer &&
          activeRect.bottom <= navRect.bottom - buffer;

    if (!isVisible) {
        const activeOffsetInScroll =
              activeRect.top - navRect.top + navWide.scrollTop;
        const stickyHeight = stickyBar ? stickyBar.offsetHeight : 0;
        const usableHeight = Math.max(0, navWide.clientHeight - stickyHeight);
        const targetTop =
              activeOffsetInScroll - stickyHeight - Math.floor(usableHeight / 3);
        navWide.scrollTop = Math.max(0, targetTop);
        nav__saveTOCScroll(navWide);
    }
}

function nav__highlightActiveTOC(navWide, tocEl) {
    const itemSelector =
          "span.chapterToc," +
          "span.appendixToc," +
          "span.sectionToc," +
          "span.subsectionToc," +
          "span.subsubsectionToc";
    const allItems = Array.from(tocEl.querySelectorAll(itemSelector));
    const currentPageItems = [];

    allItems.forEach((item) => {
        const anchor = item.querySelector("a[href]");
        if (nav__isLinkForCurrentPage(anchor)) {
            item.classList.add("toc-current-page");
            currentPageItems.push(item);
        }
    });

    if (currentPageItems.length === 0) {
        return;
    }

    let activeItem = null;
    let suppressScrollSpyUntil = 0;

    const setActiveItem = (item) => {
        if (!item || item === activeItem) {
            return;
        }
        if (activeItem) {
            activeItem.classList.remove("toc-active");
            const prevAnchor = activeItem.querySelector("a");
            if (prevAnchor) {
                prevAnchor.removeAttribute("aria-current");
            }
        }
        activeItem = item;
        activeItem.classList.add("toc-active");
        const anchor = activeItem.querySelector("a");
        if (anchor) {
            anchor.setAttribute("aria-current", "location");
        }
    };

    const findItemByHash = (hash) => {
        if (!hash) {
            return null;
        }
        return (
            currentPageItems.find((item) => {
                const anchor = item.querySelector("a[href]");
                if (!anchor) {
                    return false;
                }
                try {
                    const url = new URL(
                        anchor.getAttribute("href"),
                        window.location.href,
                    );
                    return url.hash === hash;
                } catch (e) {
                    return anchor.getAttribute("href") === hash;
                }
            }) || null
        );
    };

    const initialMatch =
          findItemByHash(window.location.hash) || currentPageItems[0];
    setActiveItem(initialMatch);

    window.addEventListener("hashchange", function () {
        const match = findItemByHash(window.location.hash);
        if (match) {
            suppressScrollSpyUntil = Date.now() + 250;
            setActiveItem(match);
            const group = match.closest(".toc-chapter-group.toc-has-children");
            if (group) {
                nav__setChapterGroupOpen(group, true);
            }
            nav__ensureActiveVisible(navWide);
        }
    });

    if (currentPageItems.length > 1) {
        const spyTargets = currentPageItems
              .map((item) => {
                  const anchor = item.querySelector("a[href]");
                  if (!anchor) {
                      return null;
                  }
                  let hash = "";
                  try {
                      hash = new URL(anchor.getAttribute("href"), window.location.href)
                          .hash;
                  } catch (e) {
                      hash = anchor.getAttribute("href") || "";
                  }
                  if (!hash || !hash.startsWith("#")) {
                      return null;
                  }
                  const targetAnchor = document.getElementById(hash.slice(1));
                  if (!targetAnchor) {
                      return null;
                  }
                  const headingEl =
                        targetAnchor.closest("h1, h2, h3, h4, h5, h6") || targetAnchor;
                  return { item, headingEl };
              })
              .filter(Boolean);

        if (spyTargets.length > 1) {
            let spyTicking = false;
            const updateScrollSpy = () => {
                spyTicking = false;
                if (Date.now() < suppressScrollSpyUntil) {
                    return;
                }

                const topOffset = 120;
                const nearBottom =
                      window.innerHeight + window.scrollY >=
                      document.documentElement.scrollHeight - 10;

                let chosen = spyTargets[0].item;
                if (nearBottom) {
                    chosen = spyTargets[spyTargets.length - 1].item;
                } else {
                    for (let i = 0; i < spyTargets.length; i++) {
                        const rect = spyTargets[i].headingEl.getBoundingClientRect();
                        if (rect.top <= topOffset) {
                            chosen = spyTargets[i].item;
                        } else {
                            break;
                        }
                    }
                }
                setActiveItem(chosen);
            };

            window.addEventListener(
                "scroll",
                function () {
                    if (!spyTicking) {
                        spyTicking = true;
                        window.requestAnimationFrame(updateScrollSpy);
                    }
                },
                { passive: true },
            );
        }
    }
}

function nav__initTOCScroll(navWide) {
    const savedScroll = nav__storageGet(TOC_SCROLL_STORAGE_KEY);
    if (savedScroll !== null) {
        const parsed = parseInt(savedScroll, 10);
        if (!Number.isNaN(parsed) && parsed >= 0) {
            navWide.scrollTop = parsed;
        }
    }

    nav__ensureActiveVisible(navWide);

    let scrollTicking = false;
    navWide.addEventListener(
        "scroll",
        function () {
            if (!scrollTicking) {
                scrollTicking = true;
                window.requestAnimationFrame(function () {
                    scrollTicking = false;
                    nav__saveTOCScroll(navWide);
                });
            }
        },
        { passive: true },
    );

    navWide.addEventListener("click", function (event) {
        if (event.target && event.target.closest("a[href]")) {
            nav__saveTOCScroll(navWide);
        }
    });

    window.addEventListener("pagehide", function () {
        nav__saveTOCScroll(navWide);
    });
}

function nav__initSidebarEnhancements() {
    const navWide = document.querySelector("nav.wide");
    if (!navWide) {
        return;
    }
    const tocEl = navWide.querySelector(".toc");
    if (!tocEl) {
        return;
    }

    nav__buildCollapsibleTOC(tocEl);
    nav__buildStickySearch(navWide, tocEl);
    nav__highlightActiveTOC(navWide, tocEl);
    nav__applySavedChapterStates(tocEl);
    nav__initTOCScroll(navWide);
}

document.addEventListener("DOMContentLoaded", function () {
    let darkMode = localStorage.getItem("darkMode") == "true";

    nav__initSidebarEnhancements();

    menu__makeCurrentPageTOC();

    //$.when(removeEmptyRows()).then(alternatingRowColors());

    FixSideMarginDerivatives();

    modeline__showLinkInformation();

    filter__removeLineswithinFloats();

    filter__repositionCaptioninMargin();

    set__Theme(darkMode);

    let button = document.querySelectorAll(".body-button");

    ShowHideBlock(button);

    filter__removeUnnecessaryTOCs();

    modeline__generateModeMenuList(moodMenu);

    // anim__highlightSideNoteImage ();

    menu__createDetailTOC();

    modeline__toggleThemeButton();

    button_codeCopy();

    //Let's do a small control where if a specific id exists for the table, we
    // turn it into a dataTable
    for (const dtbl of DataTableList) {
        if (document.getElementById(dtbl)) {
            new DataTable("#" + dtbl);
        }
    }

    // Here we insert a new function to all ellipsis events to allow sidenote
    // text to appear at the correct place.
    document.querySelectorAll('.ellipsis').forEach((el) => {
        
        el.setAttribute('onmouseenter', 'getPos(event)');

    });

});

// RESIZING FUNCTIONS ----------------------------------------------------------

$(window).resize(function () {
    button = document.querySelectorAll(".body-button");

    // Update the user on which view mode they are viewing their content on
    menu__describeState();

    if (
        util__describeViewState("Max-Computer") &&
            bool_marginImageCheck == false
    ) {
        filter__readjustMarginImages();
        bool_marginImageCheck = true;
    }

    let navWide = document.querySelector("nav.wide");

    // First determine the current state of the primary TOC.
    let tocState = getComputedStyle(navWide).display;

    if (util__describeViewState("Computer")) {
        if (tocState == "none") {
            $("nav.wide")
                .stop()
                .animate({ width: "toggle", opacity: "toggle" }, "slow");
        }
    } else {
        $("nav.wide").hide();
    }
});

// DOM-LOADED FUNCTIONS --------------------------------------------------------

$(document).ready(function () {
    // Determine the colour to be used for the topic.
    filter__colourDomainPicker();

    // Update the user on which view mode they are viewing their content on
    menu__describeState();

    // Remove the TOC if it is displayed in an unsupported view

    if (
        util__describeViewState("Max-Computer") &&
            bool_marginImageCheck == false
    ) {
        filter__readjustMarginImages();
        bool_marginImageCheck = true;
    }
});

function checkVisible(value){

    const item = value.getBoundingClientRect();
    return (
        item.top >= 0 &&
            item.left >= 0 &&
            item.bottom <= (
                window.innerHeight ||
                    document.documentElement.clientHeight) &&
            item.right <= (
                window.innerWidth ||
                    document.documentElement.clientWidth)
    );
}

function getPos(e){
    
    x=e.clientX;
    y=e.clientY;
    cursor="Your Mouse Position Is : " + x + " and " + y ;
   
    let alink = e.target;
    
    alink.nextElementSibling.style.top = x;

    // Let's define the sidenote which we want to show
    let sidenote = alink.nextElementSibling;

    console.log(sidenote);
    
    let bboxsidenote = sidenote.getBoundingClientRect()
    
    let coordinateLink = [
        alink.getBoundingClientRect().top + window.scrollY - bboxsidenote.height * 1/2,
        alink.getBoundingClientRect().left + window.scrollX - bboxsidenote.width * 3/2
        ]

    // Now let's save the coordinates of each bounding box and see whether the
    // box fits into the viewport.
    coordinates = {
        "c1" : [bboxsidenote.x, bboxsidenote.y],
        "c2" : [bboxsidenote.x + bboxsidenote.width, bboxsidenote.y],
        "c3" : [bboxsidenote.x + bboxsidenote.width, bboxsidenote.y + bboxsidenote.height],
        "c4" : [bboxsidenote.x, bboxsidenote.y + bboxsidenote.height]
    }


    let current_window = {
        "c0" : [0,0],
        "c1" : [window.innerWidth,0],
        "c2" : [window.innerWidth, window.innerHeight],
        "c3" : [0, window.innerHeight]
    }

    sidenote.style.top = coordinateLink[0] + 'px';
    sidenote.style.left = coordinateLink[1] + 'px';
    
    console.log(checkVisible(sidenote))

    
    if (checkVisible(sidenote)) {                
    } else {
        sidenote.style.top = coordinateLink[0] + 'px';
        sidenote.style.left = coordinateLink[1] - bboxsidenote.width * 3/2 + 'px';

        if (checkVisible(sidenote)) {                
        } else {

            sidenote.style.top = coordinateLink[0] - bboxsidenote.height * 3/2 + 'px';

        }
    }
}


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
            menu__toggleCurrentPageTOC();
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
