import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.SaykorNodesLoadImage.ClearButton",
    async nodeCreated(node) {
        if (node.comfyClass !== "saykor_load_image_safe") return;

        // Wait a tick for the node's widgets to be fully built
        setTimeout(() => {
            // Find the 'image' widget (the dropdown with [none] + image list)
            const imageWidget = node.widgets?.find(w => w.name === "image");
            if (!imageWidget) return;

            // Add a "Clear" button widget right after the image dropdown
            const clearBtn = node.addWidget("button", "Clear", null, () => {
                // Reset the image dropdown to "[none]"
                imageWidget.value = "[none]";

                // Trigger the widget's callback so ComfyUI internal state is updated
                if (imageWidget.callback) {
                    imageWidget.callback("[none]");
                }

                // Mark the canvas as dirty so the node re-renders
                node.setDirtyCanvas(true, false);

                // Force an immediate graph re-execution check
                if (app.graph) {
                    app.graph.change();
                }
            });

            // Style the Clear button to be slightly more visible
            clearBtn.label = "Clear";
            clearBtn.tooltip = "Reset image selection to [none]";

            // Reposition: move Clear button to be right after the image widget
            const imageIdx = node.widgets.indexOf(imageWidget);
            const btnIdx = node.widgets.indexOf(clearBtn);
            if (imageIdx !== -1 && btnIdx !== -1 && btnIdx > imageIdx) {
                // Remove from current position and insert right after image widget
                node.widgets.splice(btnIdx, 1);
                node.widgets.splice(imageIdx + 1, 0, clearBtn);
            }

            node.setDirtyCanvas(true, true);
        }, 50);
    }
});
