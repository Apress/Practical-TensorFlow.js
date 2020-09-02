const threshold = 0.7;

async function init() {
  const model = await toxicity.load(threshold);

  chrome.tabs.executeScript({
    code: 'window.getSelection().toString();',
  }, async (selection) => {
    // Get the selected text
    const selectedText = selection[0];
    document.getElementById('input').innerHTML = selectedText;
    const table = document.getElementById('predictions-table');

    await model.classify(selectedText).then((predictions) => {
      predictions.forEach((category) => {
        // Add the results to the table
        const row = table.insertRow(-1);
        const labelCell = row.insertCell(0);
        const categoryCell = row.insertCell(1);
        categoryCell.innerHTML = category.results[0].match === null ? '-' : category.results[0].match.toString();
        labelCell.innerHTML = category.label;
      });
    });
  });
}

init();
