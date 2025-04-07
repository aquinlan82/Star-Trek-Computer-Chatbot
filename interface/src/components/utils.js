import dropdowns from '../assets/dropdowns.json'

// ----------------- Functions for getting labels from codes -----------------

// Each brand has a code which is being stored for the backend and a label
// we store the label until we convert it to a code for the backend
export const label2Value = (selections) => {
    var newSelections = {...selections}
    newSelections["type"] = getTypeValue(selections["type"])
    newSelections["series"] = getSeriesValue(selections["series"])
    newSelections["model"] = getModelValue(selections["type"], selections["series"], selections["model"])
    return newSelections

}



// Type doesn't have separate codes used in the backend, just return label
export const getTypeValue = (label) => {
    return label
}

// Series doesn't have separate codes used in the backend, just return label
export const getSeriesValue = (label) => {
    return label
    
}

// get the code value for the model
export const getModelValue = (type, series, modelLabel) => {
    let temp = dropdowns[type]["children"][series]["children"]

     if (modelLabel == "All Models") {
        let arr = []
        Object.keys(temp).forEach((model_name, index) => {
            const model_id = temp[model_name].id;
            arr.push(parseInt(model_id));
        })
        return arr
    } else {
        return [parseInt(temp[modelLabel].id)];
    }
   
}

// ----------------- Functions for getting lists of options for dropdowns -----------------

// Get list of types to display based on current selections
export const getTypes = (selections) => {


    var labels = []
    Object.keys(dropdowns).forEach((type, index) => {
        labels.push(type)
    })
    return labels
}

// Get list of series to display based on current selections
export const getSeries = (selections) => {
    var type = selections["type"]

    // if brand or type hasn't been selected, return empty list
    if (type == "") {
        return []
    }

    let temp = dropdowns[type]["children"];  
    let labels = []
    Object.keys(temp).forEach((key, index) => {
        labels.push(key)
    })

    return labels

}

// Get list of models to display based on current selections
export const getModels = (selections) => {
    var type = selections["type"]
    var series = selections["series"]

    // if brand, type, or series hasn't been selected, return empty list
    if (type === "" || series === "") {
        return []
    }

    let temp = product_hierarchy[type]["children"][series]["children"];
    let labels = ["All Models"]
    Object.keys(temp).forEach((key, index) => {
        labels.push(temp[key].code)
    })

    return labels
}

// Get list of options to display based on current screen
export const getValues = (selections, screen) => {
    var functionMap = {"type": getTypes, "series": getSeries, "model": getModels}
    var labels = functionMap[screen](selections)

    var options = []
    for (let i = 0; i < labels.length; i++) {
        options.push({"label": labels[i], "value":labels[i]})
    }

    return options
}