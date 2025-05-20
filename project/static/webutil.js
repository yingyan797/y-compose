function uploadFile(id, fnameBox) {
    const segs = document.getElementById(id).value.split("\\");
    document.getElementById(fnameBox).value = segs[segs.length-1].split(".")[0];
}

function hideShow(id) {
    const elem = document.getElementById(id);
    if (elem.style.display == "none") {
        elem.style.display = "block";
    } else {
        elem.style.display = "none";
    }
}

function initKeys(query, taskEdit) {
    document.querySelectorAll(query).forEach((elem, i) => {
        elem.addEventListener("click", function() {
            inputTextAtCursor(taskEdit, elem.id.split("-")[1]);
        })
    })
}

function inputTextAtCursor(textarea, text) {
    const startPos = textarea.selectionStart;
    const endPos = textarea.selectionEnd;
    const scrollTop = textarea.scrollTop;
    
    // Get the text before and after the cursor
    const textBefore = textarea.value.substring(0, startPos);
    const textAfter = textarea.value.substring(endPos);
    
    // Special handling for parentheses
    if (text == "clear") {
        textarea.value = "";
        textarea.selectionStart = startPos;
        textarea.selectionEnd = startPos;
    } else if (text === "()") {
        // If text is selected, wrap it in parentheses
        if (startPos !== endPos) {
            const selectedText = textarea.value.substring(startPos, endPos);
            textarea.value = textBefore + "(" + selectedText + ")" + textAfter;
            textarea.selectionStart = startPos + 1;
            textarea.selectionEnd = endPos + 1;
        } else {
            // Otherwise, insert empty parentheses and place cursor between them
            textarea.value = textBefore + "() " + textAfter;
            textarea.selectionStart = startPos + 1;
            textarea.selectionEnd = startPos + 1;
        }
    } 
    // Handle all other operators
    else {
        let sep = " ";
        if (["!", "U", "F", "G"].includes(text)) {
            sep = "";
        }
        textarea.value = textBefore + text + sep + textAfter;
        const pos = startPos + text.length + sep.length;
        textarea.selectionStart = pos;
        textarea.selectionEnd = pos;
    }
    
    // Restore scroll position
    textarea.scrollTop = scrollTop;
    
    // Focus back on the textarea
    textarea.focus();
}