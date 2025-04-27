from lib.global_vars import translations, t
from app import Widget
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
from lib.products_db import ProductsDB

class ProductGridWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()

    def display(self):
        products = self.db.get_all_products()
        df = pd.DataFrame(products)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=False, flex=1)
        gb.configure_column("id", width=80)
        gb.configure_column("title", width=200)
        gb.configure_column("url", width=200)
        gb.configure_column("keywords", width=200)
        gb.configure_column("type", width=150, filter=True)
        gb.configure_column("source", width=150)
        gb.configure_column("goal", width=150)
        gb.configure_column("related", width=150)
        gb.configure_column("description", width=200, editable=True,
            cellEditor='agLargeTextCellEditor', cellEditorPopup=True, cellEditorParams={'maxLength': '50000'})
        gb.configure_column("content", width=200, editable=True,
            cellEditor='agLargeTextCellEditor', cellEditorPopup=True, cellEditorParams={'maxLength': '50000'})
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        grid_options = gb.build()
        grid_options['rowMultiSelectWithClick'] = True

        response = AgGrid(
            df,
            gridOptions=grid_options,
            height=400,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            key=f"{self.prefix}_products_grid"
        )
        return response['selected_rows']
