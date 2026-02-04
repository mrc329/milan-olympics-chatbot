# KEY CHANGES FROM ORIGINAL:
# 1. Language selector moved to TOP of sidebar (compact flag buttons)
# 2. About section made COLLAPSIBLE (collapsed by default)  
# 3. Suggestion pills KEPT (as requested)
# 4. Removed bottom language selector (was duplicate)
# 5. Compact spacing throughout for single-pane view

# Find these sections in your app.py and replace them:

# ═══════════════════════════════════════════════════════════════
# SECTION 1: TOP OF INFO COLUMN (after "with info_col:")
# Replace the entire "Competition Day / Countdown" section with this:
# ═══════════════════════════════════════════════════════════════

        # -- Language Selector (moved to top) --
        lang_options = {"EN": "🇬🇧 EN", "FR": "🇫🇷 FR", "IT": "🇮🇹 IT"}
        lang_cols = st.columns(3)
        for idx, (code, label) in enumerate(lang_options.items()):
            with lang_cols[idx]:
                is_active = (code == active_lang)
                if st.button(
                    label,
                    key=f"lang_btn_{code}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    if code != active_lang:
                        st.session_state["lang"] = code
                        st.rerun()
        
        # Small gap
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        
        # -- Competition Day / Countdown --
        if during_games:
            day_num  = (today - games_start).days + 1
            date_str = today.strftime("%A, %B %d")
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Competition Day</div>'
                f'<div class="info-day-num">Day {day_num}</div>'
                f'<div class="info-day-date">{date_str}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        elif today < games_start:
            countdown = (games_start - today).days
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Milano Cortina 2026</div>'
                f'<div class="info-day-num">{countdown} days</div>'
                f'<div class="info-day-date">Until the Games begin</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="info-day-box">'
                f'<div class="info-day-label">Milano Cortina 2026</div>'
                f'<div class="info-day-num">Finished</div>'
                f'<div class="info-day-date">Feb 6 – Feb 22, 2026</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════
# SECTION 2: REPLACE THE ABOUT SECTION
# Find "# -- About --" and replace with this collapsible version:
# ═══════════════════════════════════════════════════════════════

        # -- About (collapsible, collapsed by default) --
        with st.expander("ℹ️ About Tyler & Sasha", expanded=False):
            st.markdown(
                '<div class="about-block">'
                '<span class="about-name">Tyler</span> <span class="about-flag">USA - 2018 Bronze · Figure Skating</span><br>'
                '<span class="about-name">Sasha</span> <span class="about-flag">RUS - 2014 & 2018 Silver · Figure Skating</span>'
                '<div class="about-divider">-</div>'
                'Rivals 2014–2018. Now partners. It\'s complicated.'
                '<div class="about-stack"><strong>Stack:</strong> Pinecone · Sentence Transformers · Wikipedia</div>'
                '</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════
# SECTION 3: REMOVE THE BOTTOM LANGUAGE SECTION
# Find and DELETE these lines (near the end of info_col):
# ═══════════════════════════════════════════════════════════════

        # gap
        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

        # -- Language --
        st.markdown(f'<div class="sidebar-heading">{t("about_title") if False else "Language"}</div>', unsafe_allow_html=True)
        lang_options = {"EN": "🇬🇧 English", "FR": "🇫🇷 Français", "IT": "🇮🇹 Italiano"}
        selected = st.selectbox(
            "Language",
            options=list(lang_options.keys()),
            format_func=lambda k: lang_options[k],
            index=list(lang_options.keys()).index(active_lang),
            key="lang_select",
            label_visibility="collapsed"
        )
        if selected != active_lang:
            st.session_state["lang"] = selected
            st.rerun()

# ^^^ DELETE ALL OF THE ABOVE ^^^


# ═══════════════════════════════════════════════════════════════
# SECTION 4: COMPACT GAP SPACING
# Find all instances of this:
# ═══════════════════════════════════════════════════════════════

        st.markdown('<div class="info-section-gap"></div>', unsafe_allow_html=True)

# And replace with this SMALLER gap:

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# THAT'S IT! These 4 changes will give you:
# ═══════════════════════════════════════════════════════════════
# ✅ Language selector at TOP (compact flags)
# ✅ About section COLLAPSIBLE (saves space)
# ✅ Suggestion pills KEPT
# ✅ Single-pane layout (no scroll needed)
# ✅ No duplicate language selector at bottom
