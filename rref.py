import streamlit as st
import numpy as np
import re
import sympy as sp
import math


def splitstr(pattern, strings):
    slist = re.split(pattern, strings)
    if "" in slist:
        slist.remove("")
    return slist


def pharseString(strings):
    patternrow = r'[\n|;]'
    patterncol = r'[ |,]'
    splitRows = splitstr(patternrow, strings)
    split2d = [splitstr(patterncol, item)
               for item
               in splitRows if bool(item) if True]
    if "" in split2d:
        split2d.remove("")
    return sp.Matrix(split2d, copy=True)


def toLatex(mat, env='bmatrix', rowSep="\\\\", colSep="&",
            marked=list(), unmarked=list(), Ndigits=-1):
    temp = np.zeros((mat.shape[0], mat.shape[1])).astype(str)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            item = mat[i, j]
            if Ndigits >= 0:
                if item != 0:
                    Nint = int(np.floor(math.log(np.abs(item), 10))) + 1
                else:
                    Nint = 0
                Nsig = Ndigits + Nint
                temp[i][j] = item.evalf(Nsig)
            else:
                temp[i][j] = item
    for index in marked:
        marke = "{\\color{red}" + str(temp[index[0], index[1]]) + "}"
        temp[index[0], index[1]] = marke
    for index in unmarked:
        marke = "{\\color{black}" + str(temp[index[0], index[1]]) + "}"
        temp[index[0], index[1]] = marke
    templist = [colSep.join([str(i) for i in row]) for row in temp]
    cont = rowSep.join(templist)
    return '\\begin{' + env + '}' + cont + '\\end{' + env + '}'


def firstnonzeroelement(mat, rmod=0, cmod=0):
    nCol = mat.shape[1]
    flag = -1
    for i in range(cmod, nCol):
        nonzero = list()
        flag = 0
        for idx, entry in enumerate(mat[rmod:, i]):
            if entry != 0:
                flag = 1
                nonzero.append([idx+rmod, i])
        if flag == 1:
            break
    if flag == 1:
        res = (i, nonzero)
    elif flag == -1:
        res = (-2, list())
    else:
        res = (-1, list())
    return res


def forwardstep(inputmat, rmod=0, cmod=0):
    mat = sp.Matrix(inputmat, copy=True)
    outlog = list()
    res = firstnonzeroelement(mat, rmod=rmod, cmod=cmod)
    operations = list()
    if res[0] == -1:
        outlog.append(["markdown", "We are now looking at the matrix starting"
                       " from the Row $" + str(rmod+1) + "$ and Column $"
                       + str(cmod+1) + "$. The portion we are interested in "
                       "is marked red below."])
        markred = [[i, j] for i in range(rmod, mat.shape[0])
                   for j in range(cmod, mat.shape[1])]
        outlog.append(["latex", toLatex(mat, marked=markred)])
        outlog.append(["markdown", "All are zeros. So the process is now"
                       " stopped."])
        pivot = None
    elif res[0] == -2:
        pivot = None
    else:
        r1, c1 = res[1][0]
        pivot = (rmod, c1)
        outlog.append(["markdown", "We are now looking at the matrix starting"
                       " from the Row $" + str(rmod+1) + "$ and Column $"
                       + str(c1+1) + "$. The portion we are interested in "
                       "is marked red below."])
        markred = [[i, j] for i in range(rmod, mat.shape[0])
                   for j in range(cmod, mat.shape[1])]
        outlog.append(["latex", toLatex(mat, marked=markred)])
        outlog.append(["markdown", "We search through columns and find that"
                       " the first column in the portion that contains nonzero"
                       " numbers is Column $" + str(c1+1) + "$. "
                       "The nonzero entries are marked red. "])
        outlog.append(["latex", toLatex(mat, marked=res[1])])
        outlog.append(["markdown", "We \"*randomly*\" pick the entry $("
                       + str(r1+1) + "," + str(c1+1) + ")$ "
                       "to perform the algorithm. "])
        if r1 != rmod:
            outlog.append(["markdown", "We switch this row to the top by "
                           "switching Row $" + str(r1+1)
                           + "$ and Row $" + str(rmod+1) + "$. "])
            tempr = mat[r1, :]
            mat[r1, :] = mat[rmod, :]
            mat[rmod, :] = tempr
            outlog.append(["latex", toLatex(mat)])
            operations.append("R" + str(r1+1)
                              + "\\leftrightarrow R" + str(rmod+1)
                              + "\\Rightarrow" + toLatex(mat))
        if mat[rmod, c1] != 1:
            nc = mat[rmod, c1]
            outlog.append(["markdown", "Since this entry is not $1$, "
                           "we would like to normalize it by divide"
                           " the whole row by its entry value $"
                           + str(mat[rmod, c1]) + "$. "])
            mat[rmod, :] = mat[rmod, :] / nc
            outlog.append(["latex", toLatex(mat)])
            operations.append("R" + str(rmod+1) + "/("
                              + str(nc) + ")\\Rightarrow"
                              + toLatex(mat))
        outlog.append(["markdown", "Now we would like to make all entries "
                       "under the entry $(" + str(rmod+1) + ","
                       + str(c1+1) + ")$ to be zero by"
                       " Replacement operation. "])
        for rr, cc in res[1][1:]:
            cons = - mat[rr, cc] / mat[rmod, c1]
            outlog.append(["markdown", "- Use $R" + str(rr+1) + "+"
                           "(" + str(cons) + ")R" + str(rmod+1) + "$ "
                           "to replace $R" + str(rr+1) + "$. Look at "
                           "the entry $(" + str(rr+1) + "," + str(cc+1) +
                           ")$ and the entry $(" + str(rmod+1) + "," +
                           str(c1+1) + ")$. The constant $" + str(cons)
                           + "$ comes from $-(" + str(mat[rr, cc])
                           + ")/(" + str(mat[rmod, c1]) +
                           ")$ that this is how we change the entry $("
                           + str(rr+1) + "," + str(cc+1) + ")$ to be $0$. "
                           "The resulted matrix is"])
            mat[rr, :] = mat[rr, :] + cons * mat[rmod, :]
            outlog.append(["latex", toLatex(mat)])
            operations.append("R" + str(rr+1) + "+"
                              "(" + str(cons) + ")R" + str(rmod+1) +
                              "\\rightarrow R" + str(rr+1) +
                              "\\Rightarrow" + toLatex(mat))
        outlog.append(["markdown", "Since now all entries below the "
                       "entry $(" + str(rmod+1) + "," + str(c1+1) + ")$"
                       " are all zeros (or there are no entries below), "
                       "we stop now and turn to the next column."])
    return (outlog, operations, mat, pivot)


def forwardprocess(mat):
    # nRow, nCol = mat.shape
    outlog = list()
    ops = list([toLatex(mat)])
    totheend = False
    r, c = 0, 0
    step = 1
    pivots = list()
    while(totheend is False):
        ol, oper, mat, pivot = forwardstep(mat, rmod=r, cmod=c)
        ops.extend(oper)
        if len(ol) > 1:
            outlog.append(["markdown", "**Step " + str(step) + ":**"])
        outlog.extend(ol)
        if pivot is not None:
            r, c = pivot
            pivots.append([r, c])
            r = r + 1
            c = c + 1
            step = step + 1
        else:
            totheend = True
    outlog.append(["markdown", "**End of the forward process:**"])
    outlog.append(["markdown", "This is the end of the forward"
                   " process. The matrix is turned into an **echelon form**"
                   " or an **upper triangular form**. In the following"
                   "matrix, all the pivot positions are marked red."])
    outlog.append(["latex", toLatex(mat, marked=pivots)])
    return (outlog, ops, mat, pivots)


def backwardprocess(mat, pivots):
    outlog = list()
    operations = list()
    N = len(pivots)
    for i in range(N):
        outlog.append(["markdown", "**Step " + str(i+1)
                       + ":**"])
        r = pivots[N-i-1][0]
        c = pivots[N-i-1][1]
        outlog.append(["markdown", "We look at the pivot position $("
                       + str(r+1) + "," + str(c+1) + ")$. The pivot"
                       " and all nonzero entries above it are marked"
                       " red."])
        nz = list()
        for rr in range(r):
            if mat[rr, c] != 0:
                nz.append([rr, c])
        nzz = [[r, c]] + nz
        if bool(nz) is True:
            outlog.append(["latex", toLatex(mat, marked=nzz)])
            for item in nz:
                rc = item[0]
                cons = - mat[rc, c]
                outlog.append(["markdown", "- Use $R" + str(rc+1)
                               + "+(" + str(cons) + ")R" + str(r+1)
                               + "$ to replace $R" + str(rc+1)
                               + "$. The constant is computed using the"
                               " same idea as in the forward process."])
                mat[rc, :] = mat[rc, :] + cons * mat[r, :]
                operations.append("R" + str(rc+1) + "+(" + str(cons)
                                  + ")R" + str(r+1) + "\\rightarrow R"
                                  + str(rc+1) + "\\Rightarrow" +
                                  toLatex(mat))
                outlog.append(["latex", toLatex(mat)])
        else:
            outlog.append(["latex", toLatex(mat, marked=nzz)])
            outlog.append(["markdown", "There aren't any nonzero entries"
                           " (or no entries) above it. So we stop now and"
                           " turn to the previous pivot."])
    return (outlog, operations, mat)


def rrefprocess(mat):
    orignalmat = sp.Matrix(mat, copy=True)
    st.latex(toLatex(mat))
    st.markdown("# Forward process:")
    outlog, ops, mat, pivots = forwardprocess(mat)
    with st.expander("See detailed steps of the forward process."):
        for i in outlog:
            if i[0] == "markdown":
                st.markdown(i[1])
            if i[0] == 'latex':
                st.latex(i[1])

    st.markdown("# Backward process:")
    outlog, bops, mat = backwardprocess(mat, pivots)
    ops.extend(bops)
    with st.expander("See detailed steps of the backward process."):
        st.markdown("We only need to focus on the pivots and all the nonzero "
                    "entries above them. We start from the last pivot "
                    "and then go backwards.")
        for i in outlog:
            if i[0] == "markdown":
                st.markdown(i[1])
            if i[0] == 'latex':
                st.latex(i[1])
    st.markdown("# The reduced row echelon form:")
    st.markdown("Finally we get the reduced row echelon form (rref). "
                "The original matrix is on the left and the rref"
                " is on the right. These two matrices are called"
                " *row equivalent* to each other and use the symbol"
                "$\sim$ to denote the relation. ")
    st.markdown("The pivot positions are marked red.")
    st.latex(toLatex(orignalmat, marked=pivots)
             + "\\sim" + toLatex(mat, marked=pivots))
    st.markdown("## Here is the detailed process of all matrix operations.")
    with st.expander("See details."):
        for i in ops:
            st.latex(i)


about_info = r'''
## Dr. Xinli Xiao
Arkansas Tech University

xxiao@atu.edu

rref version: v1.0    
pivot version: v0.1
'''
menu_items = {'About': about_info}
st.set_page_config(menu_items=menu_items)


def rrefpage():
    st.title("Reduced row echelon form of a matrix")
    st.markdown("provided by Dr. Xinli Xiao, Arkansas Tech University")
    choice = st.selectbox("Please choose a way to get a matrix:",
                        options=["Manually input a matrix",
                                "Randomly generate a matrix"])

    if choice == "Manually input a matrix":
        st.markdown("# Matrix inputer")
        st.markdown("Columns are separated by spaces or `,`. "
                    "Rows are separated by new lines or `;`. ")
        inputmat = st.text_area("Please input your matrix.",
                                value="1 2\n3 4")
        mat = pharseString(inputmat)
    elif choice == "Randomly generate a matrix":
        nRow = st.number_input('Number of rows', min_value=1, value=3, step=1)
        nCol = st.number_input('Number of columns', min_value=1, value=3, step=1)
        mat = sp.randMatrix(nRow, nCol, min=-9, max=9)
        if st.button("Generate a new matrix"):
            mat = sp.randMatrix(nRow, nCol, min=-9, max=9)

    rrefprocess(mat)


def pivotpage():
    st.markdown("pivot")


page_names_to_funcs = {
    'Reduced row echelon form': rrefpage,
    'Pivot positions': pivotpage,
}


selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()