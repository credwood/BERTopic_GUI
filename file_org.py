import atexit
import logging
import os
import time
import threading
import subprocess
from collections import OrderedDict

import dearpygui.dearpygui as dpg

from data_utils import get_document_paths
from model import load_or_instantiate, fit_model, fit_and_merge, organize_by_term, save_model
from session_utils import Session, load_session, save_session

#------------- Logging -------------#

logger = logging.getLogger(__name__)
logging.basicConfig(filename='source_stream.log', level=logging.INFO)

#------------- Thread Monitoring Events and Data Structures -------------#


#------------- Session Init and Params -------------#

dpg.create_context()
if os.path.exists(f"data/session.pickle"):
    session = load_session()
else:
    session = Session()

session.LOADED_MODEL = load_or_instantiate(session.MODEL_PATH)

width, height, channels, data = dpg.load_image("images/blue_folder.png")

#------------- File Management and Display Functions -------------#
def show_saved_models():
    pass

def load_or_instantiate_helper():
    pass

def open_file(sender, app_data, user_data):
    _,  doc_path= user_data
    subprocess.call(["open", doc_path])
    
def load_database():
    docs =  session.DOCUMENTS
    for filename, doc_path in docs.items():
        if not dpg.does_item_exist(filename):
            dpg.add_button(label=filename, tag=filename, callback=open_file, width=-1,
                        height=25, user_data=[filename, doc_path], parent="doc_display")
            
            dpg.add_spacer(height=2, parent="doc_display")

def get_files(sender, app_data):
    doc_dirs = app_data['selections']
    for name, doc in doc_dirs.items():
        #print(doc_dir)
        #docs = get_document_paths(doc_dir) not needed until directory selector works
        #for doc in docs:
            #name = doc.split("/")[0]
        session.DOCUMENTS[name] = doc
    load_database()

#------------- Model Training and Inference Helper Functions -------------#

def fit_model_helper():
    results = []
    fit_thread = threading.Thread(target=fit_model, args=[list(session.DOCUMENTS.values()), list(session.DOCUMENTS.keys()), session.LOADED_MODEL, results])
    fit_thread.start()
    dpg.show_item("train_popup")
    while fit_thread.is_alive():
        time.sleep(.7)
    dpg.hide_item("train_popup")
    try:
        topics, probs, session.LOADED_MODEL = results
    except ValueError as e:
        print(e)
        logger.debug(e)
    session.FIT_TRANSFORM_RESULTS = [topics, probs]

def save_model_helper():
    session.MODEL_PATH = "model"
    save_thread = threading.Thread(target=save_model, args=[session.LOADED_MODEL, "model"])
    save_thread.start()
    dpg.show_item("save_popup")
    while save_thread.is_alive():
        time.sleep(0.7)
    dpg.hide_item("save_popup")    

def top_term_match_helper():
    if dpg.does_item_exist("result_dummy"):
        dpg.delete_item("result_dummy")
    dpg.hide_item("result_folder_org")
    dpg.add_child_window(autosize_x=True, show=True, tag="result_dummy", parent="result_term_org" )
    term = dpg.get_value("term")
    if len(term) == 0:
        dpg.show_item("input_popup")
        return
    scores, df = organize_by_term(term, session.LOADED_MODEL, session.FIT_TRANSFORM_RESULTS[0])
    docs = session.DOCUMENTS
    for name in df["Documents"]:
        doc_path = docs[name]
        dpg.add_button(label=name, tag=f"group_{name}", callback=open_file, width=-1,
                    height=25, user_data=[name, doc_path], parent="result_dummy")
            
        dpg.add_spacer(height=2, parent="result_dummy")

def cluster_select_helper(sender, data, user_data):
    df = session.FIT_TRANSFORM_RESULTS[0]
    df = df[ df["Topic"] == int(user_data) ]
    docs = session.DOCUMENTS
    for name in df["Documents"]:
        doc_path = docs[name]
        if not dpg.does_item_exist(f"group_{name}"):
            dpg.add_button(label=name, tag=f"group_{name}", callback=open_file, width=-1,
                        height=25, user_data=[name, doc_path], parent="group_doc_display")
            
            dpg.add_spacer(height=2, parent="group_doc_display")

    dpg.show_item("group_doc_display")

def cluster_display_helper():
    if dpg.does_item_exist("group_doc_dummy"):
        dpg.delete_item("group_doc_dummy")
    dpg.hide_item("result_term_org")
    dpg.add_child_window(autosize_x=True, show=True, tag="group_doc_dummy", parent="result_folder_org")
    
    for index, group in enumerate(session.FIT_TRANSFORM_RESULTS[0]["Topic"].unique()):
        if not dpg.does_item_exist(f"folder_group_{group}"):
            try:
                group_name = session.LOADED_MODEL.get_topic_info(group)["Name"][0]
            except KeyError as e:
                group_name = "Miscellaneous"
            pos = get_pos(index)
            dpg.add_image_button(label=group_name, pos=pos, tag=f"folder_group_{group}",texture_tag="folder_img", callback=cluster_select_helper, user_data=group, parent="group_doc_dummy")
            pos[1] += height+30
            dpg.add_text(group_name, parent="group_doc_dummy", pos=pos)

def get_pos(count):
    row, col = int(count//4), count%4
    x = width*col + col*150
    y = height*row + row*150
    if not col:
        x += 20
    if not row:
        y += 20
    return [x, y]

def display_results(sender, data):
    if sender == "group_org":
        cluster_display_helper()
        dpg.show_item("result_folder_org")
    else:
        top_term_match_helper()
        dpg.show_item("result_term_org")

#------------- GUI -------------#

with dpg.window(tag="main",label="window title", autosize=True):
    with dpg.menu_bar():
        with dpg.menu(label="Save Model"):
            dpg.add_menu_item(label="save", callback=save_model_helper)

    dpg.add_spacer(height=2)
    with dpg.group(horizontal=True):
        with dpg.child_window(width=400,tag="sidebar"):
            dpg.add_text("Files to organize (must load at least two hundred)")
            dpg.add_button(label="Importing info", tag="import_info", callback=lambda: dpg.show_item("info_popup"))
            dpg.add_spacer(height=2)
            dpg.add_spacer(height=5)
            with dpg.file_dialog(directory_selector=False, show=False, tag="file_dialog_tag", width=700 ,height=400, callback=get_files):
                dpg.add_file_extension(".pdf", color=(255, 255, 0, 255))
                dpg.add_file_extension(".txt", color=(255, 0, 255, 255))
                dpg.add_file_extension(".rtf", color=(255, 255, 0, 255))
                dpg.add_file_extension(".doc", color=(255, 0, 255, 255))
                dpg.add_file_extension(".docx", color=(255, 255, 0, 255))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Import Files", callback=lambda: dpg.show_item("file_dialog_tag"))
                dpg.add_button(label="Train Model", callback=fit_model_helper)
 
            dpg.add_separator()
            dpg.add_spacer(height=2)
            dpg.add_spacer(height=3)
            with dpg.child_window(autosize_x=True,tag="doc_display"):
                load_database()
                
        with dpg.window(show=False, modal=True, tag="info_popup"):
            dpg.add_text("Select a directory to get started.")
            dpg.add_text("The documents from which text can be gatered will appear in the sidebar.")
            dpg.add_text("You may start off by importing as many directories as you like, and you can update the model with more documents at anytime.")
            dpg.add_text("Note that the model requires at least a few hundred documents to generate results.")
            dpg.add_button(label="OK", callback=lambda: dpg.hide_item("info_popup"))
   
        with dpg.child_window(autosize_x=True, tag="visualizer"):
            dpg.add_text("Organize by:")
            dpg.add_spacer(height=2)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_input_text(label="Term", tag="term", width=100, height=20)
                dpg.add_button(label="Organize by term", tag="term_org", width=150, height=20, callback=display_results)
                dpg.add_text("----------------------------------------------")
                dpg.add_button(label="Organize by groups", tag="group_org", width=150, height=20, callback=display_results)

            dpg.add_spacer(height=12)
            dpg.add_child_window(autosize_x=True, show=False, tag="result_term_org")
            dpg.add_child_window(autosize_x=True, show=False, tag="result_folder_org")
        
        dpg.add_window(show=False, tag="group_doc_display")

    with dpg.window(tag="input_popup", show=False):
        dpg.add_text("Write your term in the textbox.")
        dpg.add_button(label="OK", tag="ok_pop", callback=lambda: dpg.hide_item("input_popup"))

    with dpg.window(tag="train_popup", show=False, modal=True):
        dpg.add_text("Model training in progress.")
        #dpg.add_button(label="OK", tag="ok_pop", callback=lambda: dpg.hide_item("train_popup"))

    with dpg.window(tag="save_popup", show=False, modal=True):
        dpg.add_text("Model saving in progress.")
        #dpg.add_button(label="OK", tag="ok_pop", callback=lambda: dpg.hide_item("train_popup"))

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height, default_value=data, tag="folder_img")

def safe_exit():
    save_session(session)


atexit.register(safe_exit)

dpg.create_viewport(title='File Organizer')
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main", True)
dpg.maximize_viewport()
dpg.start_dearpygui()
dpg.destroy_context()